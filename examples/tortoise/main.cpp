#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "common.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


int32_t NUM_RETURN_SEQUENCES = 4; //hardcoding this for now, analagous to "num_return_sequences arugment to inference_speech"


struct autoregressive_hparams{
    int32_t max_mel_tokens;
    int32_t max_text_tokens;
    int32_t max_conditioning_inputs;
    int32_t layers;
    int32_t model_dim;
    int32_t heads;
    int32_t number_text_tokens;
    int32_t start_text_token;
    int32_t num_embeddings;
};


//derived from ggml gpt2 reference implementation
struct gpt2_layer {
    // normalization
    struct ggml_tensor * linear_1_weights;
    struct ggml_tensor * linear_1_bias;

    struct ggml_tensor * linear_2_weights;
    struct ggml_tensor * linear_2_bias;

    // attention
    struct ggml_tensor * c_attention_attention_weights;
    struct ggml_tensor * c_attention_attention_bias;

    struct ggml_tensor * c_attention_projection_weights;
    struct ggml_tensor * c_attention_projection_bias;

    // mlp
    struct ggml_tensor * c_multi_layer_perceptron_fully_connected_weights;
    struct ggml_tensor * c_multi_layer_perceptron_fully_connected_bias;

    struct ggml_tensor * c_multi_layer_perceptron_projection_weights;
    struct ggml_tensor * c_multi_layer_perceptron_projection_bias;

};




struct autoregressive_model{
    autoregressive_hparams hparams;

    struct ggml_tensor * embedding;

    std::map<std::string, struct ggml_tensor *> tensors;

 
    struct ggml_tensor * conditioning_latent;

    struct ggml_tensor * text_embedding_weights;
    struct ggml_tensor * text_position_embedding_weights;

    struct ggml_tensor * mel_embedding_weights; 
    struct ggml_tensor * mel_position_embedding_weights;
    

    std::vector<gpt2_layer> layers;


    struct ggml_context * ctx;

    ggml_backend_buffer_t buffer_w;


    ggml_backend_t backend = NULL;



};



void save_f32_tensor(ggml_tensor * tensor, std::string path_name)
{
    std::ofstream stream;
    stream.open( path_name, std::ios::out | std::ios::binary);

    int elements = tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];

    std::vector<float> data_read( elements);
    ggml_backend_tensor_get(tensor,data_read.data(), 0 ,sizeof(float)* elements);
    stream.write(reinterpret_cast<const char*>( data_read.data() ), elements * sizeof(float));
    stream.close();
}




// derived from  gpt2_model_load(const std::string & fname, gpt2_model & model, gpt_vocab & vocab, int n_ctx, int n_gpu_layers) {
bool autoregressive_model_load(const std::string & fname, autoregressive_model & model)
{
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

      // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;


        int32_t max_mel_tokens;
        int32_t max_text_tokens;
        int32_t max_conditioning_inputs;
        int32_t layers;
        int32_t model_dim;
        int32_t heads;
        int32_t number_text_tokens;
        int32_t start_text_token;
        int32_t num_embeddings;

        fin.read((char *) &hparams.max_mel_tokens, sizeof(hparams.max_mel_tokens));
        fin.read((char *) &hparams.max_text_tokens, sizeof(hparams.max_text_tokens));
        fin.read((char *) &hparams.max_conditioning_inputs, sizeof(hparams.max_conditioning_inputs));
        fin.read((char *) &hparams.layers, sizeof(hparams.layers));
        fin.read((char *) &hparams.model_dim, sizeof(hparams.model_dim));
        fin.read((char *) &hparams.heads, sizeof(hparams.heads));
        fin.read((char *) &hparams.number_text_tokens, sizeof(hparams.number_text_tokens));
        fin.read((char *) &hparams.start_text_token, sizeof(hparams.start_text_token));
        fin.read((char *) &hparams.num_embeddings, sizeof(hparams.num_embeddings));
    
        printf("%s: max_mel_tokens = %d\n", __func__, hparams.max_mel_tokens);
        printf("%s: max_text_tokens = %d\n", __func__, hparams.max_text_tokens);
        printf("%s: max_conditioning_inputs =  %d\n", __func__, hparams.max_conditioning_inputs);
        printf("%s: layers = %d\n", __func__, hparams.layers);
        printf("%s: model_dim = %d\n", __func__, hparams.model_dim);
        printf("%s: heads = %d\n", __func__, hparams.heads);
        printf("%s: number_text_tokens =  %d\n", __func__, hparams.number_text_tokens);
        printf("%s: start_text_token =  %d\n", __func__,  hparams.start_text_token);
        printf("%s: num_embeddings =  %d\n", __func__, hparams.num_embeddings);

    
    }    

    size_t buffer_size = 0;

    buffer_size += 256 * 1024 * ggml_type_sizef(GGML_TYPE_F32); // text embedding weights

    buffer_size += 404 * 1024 * ggml_type_sizef(GGML_TYPE_F32); // text position embedding weights

    buffer_size += 1 * 1024 * ggml_type_sizef(GGML_TYPE_F32); // conditioning latent

    buffer_size +=  8194 * 1024 * ggml_type_sizef(GGML_TYPE_F32);// mel embedding weight

    buffer_size += 608 * 1024 * ggml_type_sizef(GGML_TYPE_F32); // mel position embedding weight

    for (int i = 0 ; i < 1; i ++)
    {
        //todo fix this
        buffer_size += 1024 * ggml_type_sizef(GGML_TYPE_F32); // inference model linear 1 weight
        buffer_size += 1024 * ggml_type_sizef(GGML_TYPE_F32); // inference model linear 1 bias
        
        buffer_size += 1024 * 3072 * ggml_type_sizef(GGML_TYPE_F32); // inference model attention weight
        buffer_size += 3072 * ggml_type_sizef(GGML_TYPE_F32); // inference model attention bias
        
        buffer_size += 1024 * 1024 * ggml_type_sizef(GGML_TYPE_F32); // inference model attention projection weight
        buffer_size += 1024 * ggml_type_sizef(GGML_TYPE_F32); // inference model attention projection bias
    
        buffer_size += 1024 * ggml_type_sizef(GGML_TYPE_F32); // inference model linear 2 weight
        buffer_size += 1024 * ggml_type_sizef(GGML_TYPE_F32); // inference model linear 2 bias

        buffer_size += 1024 * 4096 *  ggml_type_sizef(GGML_TYPE_F32); // inference model multi layer perceptron fully connected weight
        buffer_size += 4096 * ggml_type_sizef(GGML_TYPE_F32); // inference model multi layer perceptron fully connected bais
        
        buffer_size += 4096 * 1024 *  ggml_type_sizef(GGML_TYPE_F32); // inference model multi layer perceptron projection weight
        buffer_size += 1024 * ggml_type_sizef(GGML_TYPE_F32); // inference model multi layer perceptron projection bais

    }


    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %6.2f MB\n", __func__, buffer_size/(1024.0*1024.0));

     struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (size_t)(5 + 12),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        std::cout << "lol" << std::endl;
        model.ctx = ggml_init(params);
        std::cout << "lol2" << std::endl;

        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }


    // initialize the backend
#ifdef GGML_USE_CUBLAS
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);

        }
#endif

#ifdef GGML_USE_METAL
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
        
#endif

        if (!model.backend) {
            // fallback to CPU backend
            fprintf(stderr, "%s: using CPU backend\n", __func__);
            model.backend = ggml_backend_cpu_init();
        }

        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
            return false;
        }


        model.buffer_w = ggml_backend_alloc_buffer(model.backend, buffer_size);


        auto & ctx = model.ctx;

        model.text_embedding_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 256);
        model.text_position_embedding_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024,404);
        model.conditioning_latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024,1);
        model.mel_embedding_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024,8194);
        model.mel_position_embedding_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024,608);

        model.layers.resize(1);
        for (int i= 0; i < 1; i ++)
        {
            auto & layer = model.layers[i];

            layer.linear_1_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
            layer.linear_1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

            layer.c_attention_attention_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,3072, 1024);
            layer.c_attention_attention_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3072);

            layer.c_attention_projection_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,1024, 1024);
            layer.c_attention_projection_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

            layer.linear_2_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
            layer.linear_2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

            layer.c_multi_layer_perceptron_fully_connected_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,4096, 1024);
            layer.c_multi_layer_perceptron_fully_connected_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);


            layer.c_multi_layer_perceptron_projection_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,1024, 4096);
            layer.c_multi_layer_perceptron_projection_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);


            model.tensors["inference_model.transformer.h."+std::to_string(i)+".ln_1.weight"] = layer.linear_1_weights;
            model.tensors["inference_model.transformer.h."+std::to_string(i)+".ln_1.bias"] = layer.linear_1_bias;

            model.tensors["inference_model.transformer.h."+std::to_string(i)+".attn.c_attn.weight"] = layer.c_attention_attention_weights;
            model.tensors["inference_model.transformer.h."+std::to_string(i)+".attn.c_attn.bias"] = layer.c_attention_attention_bias;

            model.tensors["inference_model.transformer.h."+std::to_string(i)+".attn.c_proj.weight"] = layer.c_attention_projection_weights;
            model.tensors["inference_model.transformer.h."+std::to_string(i)+".attn.c_proj.bias"] = layer.c_attention_projection_bias;

            model.tensors["inference_model.transformer.h."+std::to_string(i)+".ln_2.weight"] = layer.linear_2_weights;
            model.tensors["inference_model.transformer.h."+std::to_string(i)+".ln_2.bias"] = layer.linear_2_bias;

            model.tensors["inference_model.transformer.h."+std::to_string(i)+".mlp.c_fc.weight"] = layer.c_multi_layer_perceptron_fully_connected_weights;
            model.tensors["inference_model.transformer.h."+std::to_string(i)+".mlp.c_fc.bias"] = layer.c_multi_layer_perceptron_fully_connected_bias;

            model.tensors["inference_model.transformer.h."+std::to_string(i)+".mlp.c_proj.weight"] = layer.c_multi_layer_perceptron_projection_weights;
            model.tensors["inference_model.transformer.h."+std::to_string(i)+".mlp.c_proj.bias"] = layer.c_multi_layer_perceptron_projection_bias;



        }




        //model.init_conv_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1,1024);

        //model.init_conv_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 80,1024);
        
        //model.tensors["conditioning_encoder.init.bias"] = model.init_conv_bias;
        //model.tensors["conditioning_encoder.init.weight"] = model.init_conv_weights;


        model.tensors["text_embedding.weight"] = model.text_embedding_weights;
        model.tensors["text_pos_embedding.emb.weight"] = model.text_position_embedding_weights;
        model.tensors["conditioning_latent"] = model.conditioning_latent;
        model.tensors["mel_embedding.weight"] = model.mel_embedding_weights;
        model.tensors["mel_pos_embedding.emb.weight"] = model.mel_position_embedding_weights;        

 {
        ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer_w);

        size_t total_size = 0;

        bool has_lm_head = false;

        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }
            std::cout << "made it here again " << std::endl;
            std::cout << length << std::endl;


            std::string name(length, 0);


            std::cout << "made it here again too " << std::endl;

            fin.read(&name[0], length);
            std::cout << "made it here again too " << std::endl;

            if (model.tensors.find(name) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.c_str());
                return false;
            }

            auto tensor = model.tensors[name];
            ggml_set_name(tensor, name.c_str());

            
            std::cout << ggml_nelements(tensor) << std::endl;
            std::cout <<nelements << std::endl;

            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.c_str());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.c_str(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            // for debugging
            if (1) {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.c_str(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.c_str(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }
            std::cout << "made it here" << std::endl;
            ggml_allocr_alloc(alloc, tensor);

            if (ggml_backend_is_cpu  (model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
                ) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
            } else {
                // read into a temporary buffer first, then copy to device memory
                std::cout << "made it here too " << std::endl;
                read_buf.resize(ggml_nbytes(tensor));
                fin.read(read_buf.data(), ggml_nbytes(tensor));
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
                std::cout << "??? " << std::endl;

            }

           
            total_size += ggml_nbytes(tensor);
        }

        ggml_allocr_free(alloc);
        printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    fin.close();

    return true;


}




struct ggml_cgraph * autoregressive_graph(
    const autoregressive_model & model,
    struct ggml_allocr * allocr,
    const std::vector<gpt_vocab::id> & tokens){

    const int token_count = tokens.size();


    static size_t buf_size = ggml_tensor_overhead()*95 + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);


    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_count);

    ggml_allocr_alloc(allocr, input);
    // avoid writing to tensors if we are only measuring the memory usage
    if (!ggml_allocr_is_measure(allocr)) {
        ggml_backend_tensor_set(input, tokens.data(), 0, token_count*ggml_element_size(input));
    }


    struct ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_count);
    ggml_allocr_alloc(allocr, position);
    if (!ggml_allocr_is_measure(allocr)) {
        for (int i = 0; i < token_count; ++i) {
            int32_t v = i;
            ggml_backend_tensor_set(position, &v, i*sizeof(int32_t), sizeof(v));
        }
    }


    std::cout << "reached here" << std::endl;

    struct ggml_tensor * text_embedding = ggml_get_rows(ctx0, model.text_embedding_weights,input);
    struct ggml_tensor * text_position_embedding = ggml_get_rows(ctx0, model.text_position_embedding_weights,position);


    struct ggml_tensor * reshaped_latent = ggml_reshape_4d(ctx0, model.conditioning_latent, 1,1,1,1024);

    struct ggml_tensor * embedding = ggml_add(ctx0,text_embedding, text_position_embedding);

    struct ggml_tensor * reshaped_embedding = ggml_reshape_4d(ctx0, embedding, 1,1,token_count,1024);

    struct ggml_tensor * fake_inputs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32,  token_count+2); 
    ggml_allocr_alloc(allocr, fake_inputs);
    if (!ggml_allocr_is_measure(allocr)) {
        int32_t v = 1;
        int32_t start_mel_token = 8192;
        for (int i = 0; i < token_count+1; ++i) {
            ggml_backend_tensor_set(fake_inputs, &v, i*sizeof(int32_t), sizeof(v));
        }
        ggml_backend_tensor_set(fake_inputs, &start_mel_token, (token_count+1)*sizeof(int32_t), sizeof(start_mel_token));

    }

    int32_t truncation_index = token_count + 2;

    

    struct ggml_tensor * mel_transformer_inputs =   ggml_new_tensor_1d(ctx0, GGML_TYPE_I32,4*( token_count+2));
    ggml_allocr_alloc(allocr, mel_transformer_inputs);
    
    mel_transformer_inputs = ggml_repeat(ctx0, fake_inputs, mel_transformer_inputs);
    
    mel_transformer_inputs = ggml_reshape_2d(ctx0, mel_transformer_inputs, 4, (token_count + 2)); 


    struct ggml_tensor * truncated_mel_transformer_inputs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32,4); //hardcoding this instead of slicing it from mel_transformer_inputs
    ggml_allocr_alloc(allocr, truncated_mel_transformer_inputs);
    if (!ggml_allocr_is_measure(allocr)) {
        int32_t start_mel_token = 8192;
        for (int i = 0; i < 4; ++i) {
            ggml_backend_tensor_set(truncated_mel_transformer_inputs, &start_mel_token, i*sizeof(int32_t), sizeof(start_mel_token));
        }
    }

    struct ggml_tensor * mel_embedding = ggml_get_rows(ctx0, model.mel_embedding_weights,truncated_mel_transformer_inputs);


    struct ggml_tensor * mel_position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_allocr_alloc(allocr, mel_position);
    if (!ggml_allocr_is_measure(allocr)) {
            int32_t v = 0;
            ggml_backend_tensor_set(mel_position, &v, 0, sizeof(v));
    }

    struct ggml_tensor * mel_position_embedding = ggml_get_rows(ctx0, model.mel_position_embedding_weights,mel_position);

    //struct ggml_tensor * mel_position_embedding_1d = ggml_reshape_1d(ctx0, mel_position_embedding, 1024);

    mel_embedding = ggml_add(ctx0,mel_embedding, mel_position_embedding);
 
 
    mel_embedding = ggml_reshape_4d(ctx0, mel_embedding, 1, 4, 1, 1024);


    struct ggml_tensor * output = ggml_concat(ctx0, reshaped_latent, reshaped_embedding);

    struct ggml_tensor * repeated_output = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 4 * 17 * 1024); // todo do this more cleanly, going to rely on 1d copy for same of simplicity
    output = ggml_reshape_1d(ctx0, output, 17*1024);

    //output = ggml_concat(output, )

    repeated_output =  ggml_repeat(ctx0, output, repeated_output);
    repeated_output = ggml_reshape_4d(ctx0, repeated_output, 1,4,17,1024);



    ggml_tensor * gpt2_input = ggml_concat(ctx0, repeated_output,mel_embedding);



    struct ggml_tensor * cur = ggml_reshape_4d(ctx0, gpt2_input, 1024,18,4,1);


    struct ggml_tensor * Qcur;
    struct ggml_tensor * Kcur;
    struct ggml_tensor * Vcur;
    
    struct ggml_tensor * Q;
    struct ggml_tensor * K;

    struct ggml_tensor * KQ;
    struct ggml_tensor * KQ_scaled;
    struct ggml_tensor * KQ_masked;
    struct ggml_tensor * KQ_soft_max;
    struct ggml_tensor * V_transposed;
    struct ggml_tensor * KQV;
    struct ggml_tensor * KQV_merged;

    struct ggml_tensor * residual;
    struct ggml_tensor * feed_forward_residual;

    struct ggml_tensor * test;
    for (int i = 0; i < 1; i++)
    {


           residual = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));

           //layer norm
           
           cur = ggml_norm(ctx0, cur, 1e-05);

           ggml_format_name(cur, "l%d.norm", i);
           cur = ggml_mul(ctx0, ggml_repeat(ctx0,model.layers[0].linear_1_weights, cur),cur);
           cur = ggml_add(ctx0,cur, model.layers[0].linear_1_bias);


      

            // this is implemented as conv1d in pytorch, but it's actually just a affine transformation with
            // a weight and bias
            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_attention_attention_weights)),1024,3072),
                        cur);



            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[i].c_attention_attention_bias, cur),
                    cur);
            
            cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F16,3,cur->ne));
            cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32,3,cur->ne));


            //derived from ggml reference gpt-2 implementation
            Qcur = ggml_cont(ctx0,ggml_view_3d(ctx0, cur, 1024, 18, 4, cur->nb[1], cur->nb[2], 0));
           
            Kcur = ggml_cont(ctx0,ggml_view_3d(ctx0, cur, 1024, 18, 4, cur->nb[1], cur->nb[2], 1024 * sizeof(float)));
            Vcur = ggml_cont(ctx0,ggml_view_3d(ctx0, cur, 1024, 18, 4, cur->nb[1], cur->nb[2], 2048 * sizeof(float)));

            //num heads 16
            //head dim 64
          
            Q =ggml_permute(ctx0,
                        ggml_reshape_4d(ctx0, Qcur , 64,16,18,4),
                        0, 2, 1, 3);
            
            // this is likely not general and but should work for the first layer, may need generalizing once we reach
            //the end of the first layer.
            K =ggml_permute(ctx0,
                        ggml_reshape_4d(ctx0, Kcur , 64,16,18,4),
                        0, 2, 1, 3);


            V_transposed = ggml_permute(ctx0,
                        ggml_reshape_4d(ctx0, Vcur , 64,16,18,4),
                        1,2,0,3);

            //casting to float16
            //V_transposed = ggml_cont_3d(ctx0,ggml_view_1d(ctx0,V_transposed,18*64*64,0),18, 64, 64);
            //V_transposed = ggml_reshape_4d(ctx0,ggml_cpy(ctx0, V_transposed, ggml_new_tensor(ctx0, GGML_TYPE_F16,3,V_transposed->ne)),18,64,16,4);

                    
            //casting to float16
         //   Q = ggml_cont_3d(ctx0,ggml_view_1d(ctx0,Q,64*18*64,0),64, 18, 64);
          //  Q = ggml_reshape_4d(ctx0,ggml_cpy(ctx0, Q, ggml_new_tensor(ctx0, GGML_TYPE_F16,3,Q->ne)),64,18,16,4);
           // ggml_set_name(Q, "query");
            //casting to float16
          //  K = ggml_cont_3d(ctx0,ggml_view_1d(ctx0,K,64*18*64,0),64, 18, 64);
          //  K = ggml_reshape_4d(ctx0,ggml_cpy(ctx0, K, ggml_new_tensor(ctx0, GGML_TYPE_F16,3,K->ne)),64,18,16,4);
          ///  ggml_set_name(K,"key");
        

            KQ = ggml_mul_mat(ctx0, K,Q);

            //KQ = ggml_reshape_1d(ctx0, KQ, KQ->ne[0]* KQ->ne[1]*KQ->ne[2]*KQ->ne[3]);

            //KQ = ggml_cpy(ctx0, KQ, ggml_new_tensor(ctx0, GGML_TYPE_F16,4,KQ->ne));
            //KQ = ggml_cpy(ctx0, KQ, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,KQ->ne));

            
            
            
            KQ_scaled = ggml_scale_inplace(ctx0, KQ, ggml_new_f32(ctx0,1.0f/sqrt(float(64))));


            KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, 0);


            KQ_soft_max =  ggml_soft_max_inplace(ctx0, KQ_masked);
            


            
           KQV = ggml_mul_mat(ctx0, ggml_reshape_4d(ctx0,ggml_cont(ctx0,ggml_reshape_3d(ctx0, KQ_soft_max, 18,18,64)),18,18,16,4), V_transposed);
      

           //getting the initial KQV value
           KQV = ggml_reshape_3d(ctx0, KQV, 18,64,64);
           KQV = ggml_permute(ctx0, KQV, 1,0,2,3);
           KQV = ggml_cont_3d(ctx0, KQV, 64,18,64);
           KQV = ggml_reshape_4d(ctx0, KQV, 64,18,16,4);
           

           //"merge heads" operation
           KQV_merged = ggml_permute(ctx0, KQV, 0,2,1,3);
           KQV_merged = ggml_cont_3d(ctx0, KQV_merged, 1024, 18, 4);

            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_attention_projection_weights)),1024,1024),
                        KQV_merged);



            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[i].c_attention_projection_bias, cur),
                    cur);

            //layer input passthrough
            cur = ggml_add(ctx0, cur, residual);

            feed_forward_residual = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));



            //layer norm 2
            cur = ggml_norm(ctx0, cur, 1e-05);

            ggml_format_name(cur, "l%d.norm_2", i);
            cur = ggml_mul(ctx0, ggml_repeat(ctx0,model.layers[0].linear_2_weights, cur),cur);
            cur = ggml_add(ctx0,cur, model.layers[0].linear_2_bias);



            //  fully connected multi layer perceptron
            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_multi_layer_perceptron_fully_connected_weights)),1024,4096),
                        cur);



            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[i].c_multi_layer_perceptron_fully_connected_bias, cur),
                    cur);


            cur = ggml_gelu(ctx0, cur);


            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_multi_layer_perceptron_projection_weights)),4096,1024),
                        cur);


            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[i].c_multi_layer_perceptron_projection_bias, cur),
                    cur);


            cur = ggml_add(ctx0, cur, feed_forward_residual);


    }





    std::cout << "didn't reach here" << std::endl;

    ggml_build_forward_expand(gf, cur);

    std::cout << "reached end graph build" << std::endl;

    ggml_free(ctx0);


    return gf;
    
}


int main(int argc, char ** argv) {

    std::cout << "hello world" << std::endl;
    

    gpt_vocab vocab;
    gpt_vocab_init("../examples/tortoise/tokenizer.json", vocab);
    
    std::string message = "this[SPACE]is[SPACE]a[SPACE]test[SPACE]message";
    //std::vector<gpt_vocab::id> tokens = ::gpt_tokenize(vocab, message);
    std::vector<gpt_vocab::id> tokens = ::parse_tokens_from_string("255,147,2,54,2,14,2,33,218,2,26,61,150,112,0,0", ','); // for now, skipping some token processing steps
    //std::vector<gpt_vocab::id> tokens = ::parse_tokens_from_string("255,147,2,54,2,14,2,33,218,2,26,61,150,112,0,0", ','); // for now, skipping some token processing steps


    for (int i =0; i < tokens.size(); i ++)
    {
        std::cout << tokens[i] << std::endl;
    }   
    //todo see why this tokenization doesn't match the tokenization produced by tortoise-tts (tortoise tts one does not always use the token corresponding to the most characters)


    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();


    int64_t t_load_us = 0;

    std::string file_path = "../examples/tortoise/ggml-model.bin";


    autoregressive_model model;



    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!autoregressive_model_load(file_path, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, file_path.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;

    }

    std::cout << "completed" << std::endl;
    

    ggml_backend_buffer_t buf_compute;

    struct ggml_allocr * allocr = NULL;
    // allocate the compute buffer
    {
         // alignment required by the backend
        size_t align = ggml_backend_get_alignment(model.backend);
        std::cout << "alignment" << std::endl;
        allocr = ggml_allocr_new_measure(align);
        std::cout << "align created" << std::endl;

        // create the worst case graph for memory usage estimation
        //int n_tokens = std::min(model.hparams.n_ctx, params.n_batch);
        //int n_past = model.hparams.n_ctx - n_tokens;
        ggml_allocr_reset(allocr);
        struct ggml_cgraph * gf = autoregressive_graph(model, allocr, tokens);
        ggml_graph_print(gf);

        std::cout << "graph created" << std::endl;
        // compute the required memory
        size_t mem_size = ggml_allocr_alloc_graph(allocr, gf);

        // recreate the allocator with the required memory
        ggml_allocr_reset(allocr);
        buf_compute = ggml_backend_alloc_buffer(model.backend, mem_size);
        allocr = ggml_allocr_new_from_buffer(buf_compute);
        gf = autoregressive_graph(model, allocr, tokens);
        ggml_allocr_alloc_graph(allocr, gf);
        std::cout << "reached computing time" << std::endl;
        ggml_backend_graph_compute(model.backend, gf);
        ggml_graph_print(gf);


        std::cout << "reaced end" << std::endl;
        for (int i =0; i < gf->n_nodes; i ++)
        {
            std::cout << "---------------------------------------------------" << std::endl;
            ggml_tensor * test = gf->nodes[i];
            std::cout << "NAME:" << std::endl;
            std::cout << test->name << std::endl;
            std::cout << "TYPE" << std::endl;
            std::cout <<  test->type << std::endl;
            std::cout << "SHAPE:" << std::endl;
            std::cout << test->ne[0]<< std::endl;
            std::cout << test->ne[1]<< std::endl;
            std::cout << test->ne[2]<< std::endl;
            std::cout << test->ne[3]<< std::endl;
            std::cout << "DATA:" << std::endl;


            if (!ggml_is_contiguous(test))
            {
                std::cout << "Skipped data; not contiguous" << std::endl;
                continue;
            }
            if (ggml_is_transposed(test))
            {
                std::cout << "Transposed:" << std::endl;

            }

            int elements = test->ne[0] * test->ne[1] * test->ne[2] * test->ne[3];

            //ggml_tensor * weights = gf->leafs[gf->n_leafs -2];
            //ggml_tensor * tokens = gf->leafs[gf->n_leafs -1];

            //ggml_graph_dump_dot(gf, NULL, "autoregressive.dot");
            //std::cout << "made it here" << std::endl;
            if (test->type == GGML_TYPE_F32)
            {
            std::vector<float> test_read( elements);
            ggml_backend_tensor_get(test,test_read.data(), 0 ,sizeof(float)* elements);
//        
            for (int c = 0; c < elements ; c++)
            {
                 
                if  (c < 3 || c > elements-4  || c == 1024*18-1|| c == 1024*18-2|| c == 1024*18 || c == 1024*18+2 )
                {
               
                std::cout << (test_read.data()[c])<< std::endl;
                }
            }
            }
            else if(test->type == GGML_TYPE_F16)
            {
             std::vector<ggml_fp16_t> test_read( elements);
            ggml_backend_tensor_get(test,test_read.data(), 0 ,sizeof(ggml_fp16_t)* elements);
//        
            for (int c = 0; c < elements ; c++)
            {
                if  (c < 3 || c > elements-4)
                {
                
                std::cout << ggml_fp16_to_fp32(test_read.data()[c])<< std::endl;
                }
            }
            }
        
        
        
        }

        ggml_graph_print   (gf);


        //std::cout << (float * )test->data << std::endl;
        //std::cout <<"test" << std::endl;
        //std::cout << ggml_get_i32_1d(test,0) << std::endl;



        //fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0/1024.0);
        
    }

    return 0;
  
}
