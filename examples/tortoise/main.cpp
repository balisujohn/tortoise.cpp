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

#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <functional>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


int32_t NUM_RETURN_SEQUENCES = 4; //hardcoding this for now, analagous to "num_return_sequences arugment to inference_speech"

std::mt19937 generator(245645656);
std::uniform_real_distribution<float> distribution(0.0, 1.0);

void localAssert(bool condition)
{
    if (!condition)
    {
        std::cout << "failure" << std::endl;
        exit(0);

    }
}

/*
 
 ██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗████████╗███████╗██████╗     
 ██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝╚══██╔══╝██╔════╝██╔══██╗    
 ███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝██████╔╝███████║██████╔╝███████║██╔████╔██║█████╗     ██║   █████╗  ██████╔╝    
 ██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝     ██║   ██╔══╝  ██╔══██╗    
 ██║  ██║   ██║   ██║     ███████╗██║  ██║██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║███████╗   ██║   ███████╗██║  ██║    
 ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝    
                                                                                                                          
 ███╗   ███╗ █████╗ ███╗   ██╗██╗███████╗███████╗███████╗████████╗                                                        
 ████╗ ████║██╔══██╗████╗  ██║██║██╔════╝██╔════╝██╔════╝╚══██╔══╝                                                        
 ██╔████╔██║███████║██╔██╗ ██║██║█████╗  █████╗  ███████╗   ██║                                                           
 ██║╚██╔╝██║██╔══██║██║╚██╗██║██║██╔══╝  ██╔══╝  ╚════██║   ██║                                                           
 ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║██║     ███████╗███████║   ██║                                                           
 ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝     ╚══════╝╚══════╝   ╚═╝                                                           
                                                                                                                          
 
*/

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


/*
 
  ██████╗ ██████╗ ████████╗   ██████╗                             
 ██╔════╝ ██╔══██╗╚══██╔══╝   ╚════██╗                            
 ██║  ███╗██████╔╝   ██║█████╗ █████╔╝                            
 ██║   ██║██╔═══╝    ██║╚════╝██╔═══╝                             
 ╚██████╔╝██║        ██║      ███████╗                            
  ╚═════╝ ╚═╝        ╚═╝      ╚══════╝                            
                                                                  
 ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗              
 ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗             
    ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝             
    ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗             
    ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║             
    ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝             
                                                                  
 ███╗   ███╗ █████╗ ███╗   ██╗██╗███████╗███████╗███████╗████████╗
 ████╗ ████║██╔══██╗████╗  ██║██║██╔════╝██╔════╝██╔════╝╚══██╔══╝
 ██╔████╔██║███████║██╔██╗ ██║██║█████╗  █████╗  ███████╗   ██║   
 ██║╚██╔╝██║██╔══██║██║╚██╗██║██║██╔══╝  ██╔══╝  ╚════██║   ██║   
 ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║██║     ███████╗███████║   ██║   
 ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝     ╚══════╝╚══════╝   ╚═╝   
                                                                  
 
*/


//derived from ggml gpt2 reference implementation
struct gpt2_layer {
    // layer norm 1 and 2, accidentally named incorrectly
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

    struct ggml_tensor * final_layer_norm_weights;
    struct ggml_tensor * final_layer_norm_bias;

    struct ggml_tensor * language_model_head_layer_norm_weights;
    struct ggml_tensor * language_model_head_layer_norm_bias;

    struct ggml_tensor * language_model_head_linear_weights;
    struct ggml_tensor * language_model_head_linear_bias;

    struct ggml_tensor * memory_key; 
    struct ggml_tensor * memory_value;

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


void extract_latents_to_vector(ggml_tensor * tensor, std::vector<float>  & latents)
{
    int elements = tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
    latents.resize(elements);
    ggml_backend_tensor_get(tensor,latents.data(), 0 ,sizeof(float)* elements);
}


/*
 
  ██████╗ ██████╗ ████████╗   ██████╗                
 ██╔════╝ ██╔══██╗╚══██╔══╝   ╚════██╗               
 ██║  ███╗██████╔╝   ██║█████╗ █████╔╝               
 ██║   ██║██╔═══╝    ██║╚════╝██╔═══╝                
 ╚██████╔╝██║        ██║      ███████╗               
  ╚═════╝ ╚═╝        ╚═╝      ╚══════╝               
                                                     
 ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ 
 ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗
    ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝
    ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗
    ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║
    ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
                                                     
 ██╗      ██████╗  █████╗ ██████╗                    
 ██║     ██╔═══██╗██╔══██╗██╔══██╗                   
 ██║     ██║   ██║███████║██║  ██║                   
 ██║     ██║   ██║██╔══██║██║  ██║                   
 ███████╗╚██████╔╝██║  ██║██████╔╝                   
 ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝                    
  
*/
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

    for (int i = 0 ; i < 30; i ++)
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

    buffer_size += 404 * 30 * ggml_type_sizef(GGML_TYPE_F32) *1024 * 4; // key cache (memory_key)
    buffer_size +=  404 * 30 * ggml_type_sizef(GGML_TYPE_F32) * 1024 * 4; // value cache (memory_value)
    

    buffer_size += 1024 * ggml_type_sizef(GGML_TYPE_F32); // final layer norm weight
    buffer_size += 1024 * ggml_type_sizef(GGML_TYPE_F32); // final layer norm bias

    buffer_size += 1024 * ggml_type_sizef(GGML_TYPE_F32); // language model head layer norm weight
    buffer_size += 1024 * ggml_type_sizef(GGML_TYPE_F32); // language model head layer norm bias

    buffer_size += 1024 * 8194  * ggml_type_sizef(GGML_TYPE_F32); // language model head linear weight
    buffer_size += 8194 * ggml_type_sizef(GGML_TYPE_F32); // language model head linear bias

    buffer_size += 128; // related to alignment


    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %6.2f MB\n", __func__, buffer_size/(1024.0*1024.0));

     struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (size_t)(5 + 12*30 + 8),
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
        std::cout << "created backend" << std::endl;
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
        model.final_layer_norm_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
        model.final_layer_norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
        model.language_model_head_layer_norm_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
        model.language_model_head_layer_norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
        model.language_model_head_linear_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024,8194);
        model.language_model_head_linear_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8194);


        model.layers.resize(1);
        for (int i= 0; i < 30; i ++)
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
        model.tensors["inference_model.lm_head.0.weight"] = model.language_model_head_layer_norm_weights;
        model.tensors["inference_model.lm_head.0.bias"] = model.language_model_head_layer_norm_bias;

        model.tensors["inference_model.lm_head.1.weight"] = model.language_model_head_linear_weights;
        model.tensors["inference_model.lm_head.1.bias"] = model.language_model_head_linear_bias;

        model.tensors["inference_model.transformer.ln_f.weight"] = model.final_layer_norm_weights;
        model.tensors["inference_model.transformer.ln_f.bias"] = model.final_layer_norm_bias;
        model.tensors["text_embedding.weight"] = model.text_embedding_weights;
        model.tensors["text_pos_embedding.emb.weight"] = model.text_position_embedding_weights;
        model.tensors["conditioning_latent"] = model.conditioning_latent;
        model.tensors["mel_embedding.weight"] = model.mel_embedding_weights;
        model.tensors["mel_pos_embedding.emb.weight"] = model.mel_position_embedding_weights;    


        model.memory_key = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * 404 * 30 * 1024);
        model.memory_value = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 *  404 * 30 * 1024);    
        

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


        ggml_allocr_alloc(alloc, model.memory_key);
        ggml_allocr_alloc(alloc, model.memory_value);


        ggml_allocr_free(alloc);
        printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    fin.close();

    return true;


}

/*
                                                                                                                                                                                            
  ██████╗ ██████╗ ████████╗   ██████╗     ██╗      █████╗ ████████╗███████╗███╗   ██╗████████╗    ███████╗ ██████╗ ██████╗ ██╗    ██╗ █████╗ ██████╗ ██████╗     ██████╗  █████╗ ███████╗███████╗
 ██╔════╝ ██╔══██╗╚══██╔══╝   ╚════██╗    ██║     ██╔══██╗╚══██╔══╝██╔════╝████╗  ██║╚══██╔══╝    ██╔════╝██╔═══██╗██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔══██╗    ██╔══██╗██╔══██╗██╔════╝██╔════╝
 ██║  ███╗██████╔╝   ██║█████╗ █████╔╝    ██║     ███████║   ██║   █████╗  ██╔██╗ ██║   ██║       █████╗  ██║   ██║██████╔╝██║ █╗ ██║███████║██████╔╝██║  ██║    ██████╔╝███████║███████╗███████╗
 ██║   ██║██╔═══╝    ██║╚════╝██╔═══╝     ██║     ██╔══██║   ██║   ██╔══╝  ██║╚██╗██║   ██║       ██╔══╝  ██║   ██║██╔══██╗██║███╗██║██╔══██║██╔══██╗██║  ██║    ██╔═══╝ ██╔══██║╚════██║╚════██║
 ╚██████╔╝██║        ██║      ███████╗    ███████╗██║  ██║   ██║   ███████╗██║ ╚████║   ██║       ██║     ╚██████╔╝██║  ██║╚███╔███╔╝██║  ██║██║  ██║██████╔╝    ██║     ██║  ██║███████║███████║
  ╚═════╝ ╚═╝        ╚═╝      ╚══════╝    ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝     ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝
                                                                                                                                                                                                                                                                                                                                                                 
*/
struct ggml_cgraph * autoregressive_latent_graph(
    const autoregressive_model & model,
    struct ggml_allocr * allocr,
    const std::vector<int>  mel_transformer_inputs_vector,
    const std::vector<gpt_vocab::id> & tokens
){
    const int token_count = tokens.size();
    //std::cout << "token count: " << token_count << std::endl;
    //exit(0);
    const int mel_token_count = mel_transformer_inputs_vector.size();



    static size_t buf_size = ggml_tensor_overhead()*(25 +70*38)  + ggml_graph_overhead(); //placeholder big value 
    static std::vector<uint8_t> buf(buf_size);


    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);


    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_count * 4);


    std::cout << "token count" << token_count << std::endl;

    ggml_allocr_alloc(allocr, input);
    // avoid writing to tensors if we are only measuring the memory usage
    if (!ggml_allocr_is_measure(allocr)) {
        for (int i =0; i < 4; i ++)
        {
            ggml_backend_tensor_set(input, tokens.data(), i * token_count * ggml_element_size(input) , token_count*ggml_element_size(input));
        }
    }

    ggml_set_name(input, "text input codes");



   struct ggml_tensor * input_position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_count * 4);
    ggml_allocr_alloc(allocr, input_position);
    if (!ggml_allocr_is_measure(allocr)) {
            for (int i = 0; i < 4; i ++)
            {
                for (int c =0; c < token_count; c++)
                {
                int32_t v = c;
                ggml_backend_tensor_set(input_position, &v, ((i * token_count)+c) * sizeof(int32_t), sizeof(v));
                }
            
            }
    }




    struct ggml_tensor * mel_codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, mel_token_count);

    ggml_set_name(mel_codes, "mel_codes");

    ggml_allocr_alloc(allocr, mel_codes);

    if (!ggml_allocr_is_measure(allocr)) {
        for (int i = 0; i < mel_token_count; ++i) {
            int32_t v = mel_transformer_inputs_vector[i];
            ggml_backend_tensor_set(mel_codes, &v, i*sizeof(int32_t), sizeof(v));
        }
    }

   // ggml_tensor * temp_cur = ggml_cpy(ctx0, mel_codes, ggml_new_tensor(ctx0, GGML_TYPE_I32,4,mel_codes->ne) );

    struct ggml_tensor * mel_embedding = ggml_get_rows(ctx0, model.mel_embedding_weights,mel_codes);


    //struct ggml_tensor * mel_embedding = ggml_get_rows(ctx0, model.mel_embedding_weights,input);

    ggml_set_name(model.mel_embedding_weights, "mel emb weights");
    ggml_set_name(mel_embedding, "mel embedding");



    struct ggml_tensor * mel_position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, mel_token_count);
    ggml_allocr_alloc(allocr, mel_position);
    if (!ggml_allocr_is_measure(allocr)) {
            for (int i = 0; i < 4; i ++)
            {
                for (int c =0; c < mel_token_count / 4; c++)
                {
                int32_t v = c;
                ggml_backend_tensor_set(mel_position, &v, ((i * (mel_token_count/4))+c) * sizeof(int32_t), sizeof(v));
                }
            
            }
    }

    struct ggml_tensor * mel_position_embedding = ggml_get_rows(ctx0, model.mel_position_embedding_weights,mel_position);




    ggml_set_name(mel_position_embedding, "mel position embedding");




    mel_embedding = ggml_add(ctx0,mel_embedding, mel_position_embedding);



    struct ggml_tensor * text_embedding = ggml_get_rows(ctx0, model.text_embedding_weights,input);
    struct ggml_tensor * text_position_embedding = ggml_get_rows(ctx0, model.text_position_embedding_weights,input_position);



    struct ggml_tensor * embedding = ggml_add(ctx0,text_embedding, text_position_embedding);
    ggml_set_name(embedding, "final text embedding");

    
    
    embedding =   ggml_reshape_4d(ctx0, embedding, 1,4,token_count,1024);
    mel_embedding =   ggml_reshape_4d(ctx0, mel_embedding, 1,4,mel_token_count/4,1024);


    
    
    
    struct ggml_tensor * output = ggml_concat(ctx0, embedding, mel_embedding);


    struct ggml_tensor * reshaped_latent = ggml_reshape_1d(ctx0, model.conditioning_latent, 1024);


    struct ggml_tensor * repeated_latent = ggml_new_tensor_1d(ctx0,GGML_TYPE_F32, 4 * 1024);

    ggml_allocr_alloc(allocr, repeated_latent);


    //repeated_latent = ggml_reshape_4d(ctx0, repeated_latent,1024, 1, 4, 1);

    std::cout << "repeated latent shape" << std::endl;
    std::cout << repeated_latent->ne[0] << std::endl;
    std::cout << repeated_latent->ne[1] << std::endl;
    std::cout << repeated_latent->ne[2] << std::endl;
    std::cout << repeated_latent->ne[3] << std::endl;


    std::cout << "reshaped latent shape" << std::endl;
    std::cout << reshaped_latent->ne[0] << std::endl;
    std::cout << reshaped_latent->ne[1] << std::endl;
    std::cout << reshaped_latent->ne[2] << std::endl;
    std::cout << reshaped_latent->ne[3] << std::endl;



    repeated_latent = ggml_repeat(ctx0, reshaped_latent, repeated_latent);

    repeated_latent = ggml_reshape_4d(ctx0, repeated_latent, 1,4,1,1024);



    output = ggml_concat(ctx0, repeated_latent, output);






    ggml_set_name(output, "output");

    int test_dimension = output->ne[2];
    int n_past = 0;

    std::cout << "test dimension: " << test_dimension << std::endl;

    struct ggml_tensor * cur = ggml_reshape_4d(ctx0, output, 1024,test_dimension,4,1); // this nonsense is necessary because when I modified ggml_concat I accidentall flipped the order of the dimensions it expects. This will need to be cleaned up later.


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

    for (int i = 0; i < 30; i++)
    {
           

           residual = ggml_cpy(ctx0, cur, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,1024,test_dimension,4,1));
           
           //ggml_build_forward_expand(gf, residual);
           //layer norm
           
           cur = ggml_norm(ctx0, cur, 1e-05);

           ggml_tensor * temp_cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne) );


           ggml_set_name(temp_cur, "postnorm");
           ggml_build_forward_expand(gf, temp_cur);
            

           ggml_format_name(cur, "l%d.norm", i);

           ggml_tensor * temp_ln_1_weights = ggml_repeat(ctx0,model.layers[i].linear_1_weights, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));

           ggml_set_name(temp_ln_1_weights, "weights");
           //ggml_build_forward_expand(gf, temp_ln_1_weights);


           cur = ggml_mul(ctx0, cur,temp_ln_1_weights); // if you flip the order of this it doesn't work on the second token generation process.TODO why does flipping the order of this break it?
            
           
           cur = ggml_add(ctx0,cur, model.layers[i].linear_1_bias);
           
          


            //ggml_tensor * temp_weights = ggml_cpy(ctx0, model.layers[i].linear_1_weights, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,model.layers[i].linear_1_weights->ne) );
            ggml_tensor * temp_bias = ggml_cpy(ctx0, model.layers[i].linear_1_bias, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,model.layers[i].linear_1_bias->ne) );
 
          //  if(!fake_inputs)
           // {
            //ggml_build_forward_expand(gf, temp_bias);
            //ggml_set_name(temp_bias, "bias");
           // return gf;
           // } 

                
          
            // this is implemented as conv1d in pytorch, but it's actually just a affine transformation with
            // a weight and bias
            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_attention_attention_weights)),1024,3072),
                        cur);


            cur = ggml_reshape_4d(ctx0, cur, 3072,test_dimension,4,1);


            cur = ggml_add(ctx0,cur,
                    model.layers[i].c_attention_attention_bias); // worth studying the diffs here, why did I have to remove this repeat for settings where the second dimension is 1?
            
            
            cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F16,3,cur->ne));
            cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32,3,cur->ne));

            

         



            //derived from ggml reference gpt-2 implementation
            Qcur = ggml_cont(ctx0,ggml_view_3d(ctx0, cur, 1024, test_dimension, 4, cur->nb[1], cur->nb[2], 0));
           

            //Kcur = ggml_cont(ctx0,ggml_permute(ctx0,ggml_view_4d(ctx0, cur, 1024, test_dimension, 4, 1,cur->nb[1], cur->nb[2],cur->nb[3], 1024 * sizeof(float)),0,1,3,2));

            Kcur = ggml_cont(ctx0,ggml_permute(ctx0,ggml_view_3d(ctx0, cur, 1024, test_dimension, 4, cur->nb[1], cur->nb[2], 1024 * sizeof(float)),0,2,1,3));
            Vcur = ggml_cont(ctx0,ggml_permute(ctx0,ggml_view_3d(ctx0, cur, 1024, test_dimension, 4, cur->nb[1], cur->nb[2], 2048 * sizeof(float)),0,2,1,3));

          //  struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_key, 1024 * test_dimension * 4 , (ggml_element_size(model.memory_key)* ((i * 404 * 1024 * 4) + ((n_past) * 1024 *4))));
          //  ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
            
          //  struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_value, 1024 * test_dimension * 4 , (ggml_element_size(model.memory_value)* ((i * 404 * 1024 * 4) + ((n_past) * 1024 *4))));
          //  ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));




            //num heads 16
            //head dim 64
          
            Q =ggml_cont(ctx0,ggml_permute(ctx0,
                        ggml_reshape_4d(ctx0, Qcur , 64,16,test_dimension,4),
                        0, 2, 1, 3));

            
            // this is likely not general and but should work for the first layer, may need generalizing once we reach
            //the end of the first layer.
           // K =ggml_cont(ctx0,ggml_permute(ctx0,
           //             ggml_reshape_4d(ctx0, ggml_view_1d(ctx0, model.memory_key, 1024 * (test_dimension + n_past) * 4 , (ggml_element_size(model.memory_key)* (i * 404 * 1024 * 4) )) , 64,16, 4 , test_dimension + n_past),
           //             0, 2,3, 1));
            K =ggml_cont(ctx0,ggml_permute(ctx0,
                        ggml_reshape_4d(ctx0, Kcur , 64,16,4,test_dimension),
                        0, 2, 3, 1));

 
            V_transposed =ggml_cont(ctx0,ggml_permute(ctx0,
            ggml_reshape_4d(ctx0, Vcur , 64,16,4,test_dimension),
            1, 2, 3, 0));
 
          


         //   V_transposed = ggml_cont(ctx0,ggml_permute(ctx0,
         //               ggml_reshape_4d(ctx0, ggml_view_1d(ctx0, model.memory_value, 1024 * (test_dimension + n_past) * 4 , (ggml_element_size(model.memory_value)* (i * 404 * 1024 * 4) )) , 64,16,4,test_dimension+n_past),
         //               1,2,3,0));

          

            //std::cout << K->ne[0]  << K->ne[1] << K->ne[2]  << K->ne[3] << std::endl;
            //std::cout << Q->ne[0]  << Q->ne[1] << Q->ne[2]  << Q->ne[3] << std::endl;
            //std::cout << V_transposed->ne[0]  << V_transposed->ne[1] << V_transposed->ne[2]  << V_transposed->ne[3] << std::endl;
            KQ = ggml_mul_mat(ctx0, K,Q);

            //std::cout << "KQ shape" << std::endl;
            //std::cout << KQ->ne[0]  << KQ->ne[1] << KQ->ne[2]  << KQ->ne[3] << std::endl;

        
                   std::cout << "reached: " << i << std::endl;

          
            

            //KQ = ggml_reshape_1d(ctx0, KQ, KQ->ne[0]* KQ->ne[1]*KQ->ne[2]*KQ->ne[3]);

            //KQ = ggml_cpy(ctx0, KQ, ggml_new_tensor(ctx0, GGML_TYPE_F16,4,KQ->ne));
            //KQ = ggml_cpy(ctx0, KQ, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,KQ->ne));

            
            
            
            KQ_scaled = ggml_scale_inplace(ctx0, KQ, ggml_new_f32(ctx0,1.0f/sqrt(float(64))));


            KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);


            KQ_soft_max =  ggml_soft_max_inplace(ctx0, KQ_masked);
            

            KQV = ggml_mul_mat(ctx0, KQ_soft_max, V_transposed);
            
           //KQV = ggml_mul_mat(ctx0, ggml_reshape_4d(ctx0,ggml_cont(ctx0,ggml_reshape_3d(ctx0, KQ_soft_max, test_dimension + n_past,test_dimension + n_past,64)),test_dimension + n_past,test_dimension+n_past,16,4), ggml_reshape_3d(ctx0,V_transposed,n_past + test_dimension, 16,);
      
            ggml_set_name(KQ_soft_max, "after KQ");
            ggml_set_name(V_transposed, "after V");


          
      
        

           //getting the initial KQV value
           KQV = ggml_reshape_3d(ctx0, KQV, test_dimension ,64,64);
           KQV = ggml_permute(ctx0, KQV, 1,0,2,3);
           KQV = ggml_cont_3d(ctx0, KQV, 64,test_dimension,64);
           KQV = ggml_reshape_4d(ctx0, KQV, 64,test_dimension,16,4);
           

           //"merge heads" operation
           KQV_merged = ggml_permute(ctx0, KQV, 0,2,1,3);
           KQV_merged = ggml_cont_3d(ctx0, KQV_merged, 1024, test_dimension, 4);
    

          
      



            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_attention_projection_weights)),1024,1024),
                        KQV_merged);



            cur = ggml_add(ctx0,cur,
                   model.layers[i].c_attention_projection_bias);




            //layer input passthrough
            //cur = ggml_add(ctx0, cur,residual);

            cur = ggml_add(ctx0, ggml_reshape_1d(ctx0,cur, 1024 * 4 * test_dimension), ggml_reshape_1d(ctx0,residual, 1024 * 4 * test_dimension)); // it's really strange that this is necessary, why does it have different behavior than commented out above 
            cur = ggml_reshape_4d(ctx0, cur, 1024, test_dimension, 4, 1);
    


            




            feed_forward_residual = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));




            //layer norm 2
            cur = ggml_norm(ctx0, cur, 1e-05);

            ggml_format_name(cur, "l%d.norm_2", i);

            ggml_tensor * temp_ln_2_weights = ggml_repeat(ctx0,model.layers[i].linear_2_weights, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));

            ggml_set_name(temp_ln_2_weights, "test");


            ggml_build_forward_expand(gf,temp_ln_2_weights);

            cur = ggml_mul(ctx0, cur, temp_ln_2_weights);
            cur = ggml_add(ctx0,cur, model.layers[i].linear_2_bias);




            //  fully connected multi layer perceptron
            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_multi_layer_perceptron_fully_connected_weights)),1024,4096),
                        cur);



            cur = ggml_add(ctx0,cur, model.layers[i].c_multi_layer_perceptron_fully_connected_bias);

            // gelu
            cur = ggml_gelu(ctx0, cur);


            // mlp fully connected
            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_multi_layer_perceptron_projection_weights)),4096,1024),
                        cur);


            cur = ggml_add(ctx0, cur, model.layers[i].c_multi_layer_perceptron_projection_bias);



            //final residual
            //another case where I had to flatten before adding to get correct results. Either ggml had a pre-existing bug for batch addition, or one of my modifications introduced it. This will need to be addressed.
            
            cur = ggml_add(ctx0, ggml_reshape_1d(ctx0,cur , 1024 *  test_dimension *  4 *  1) , ggml_reshape_1d(ctx0,feed_forward_residual, 1024 *  test_dimension *  4 *  1));


            cur = ggml_reshape_4d(ctx0, cur, 1024, test_dimension, 4, 1);
            //cur = ggml_add(ctx0, cur, feed_forward_residual);


            
        


    }



    cur = ggml_norm(ctx0, cur, 1e-05);


    ggml_tensor * temp_final_layer_norm_weights = ggml_repeat(ctx0,model.final_layer_norm_weights, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));



    ggml_build_forward_expand(gf,temp_final_layer_norm_weights);



    cur = ggml_mul(ctx0, cur, temp_final_layer_norm_weights);
    cur = ggml_add(ctx0,cur, model.final_layer_norm_bias);


    
    
    
    cur = ggml_norm(ctx0, cur, 1e-05);

    cur = ggml_cont(ctx0,ggml_view_4d(ctx0, cur, 1024, test_dimension-1, 4, 1, cur->nb[1], cur->nb[2], cur->nb[3], (1) * sizeof(float) * 1024 )); 



    ggml_tensor * temp_language_model_head_layer_norm_weights = ggml_repeat(ctx0,model.language_model_head_layer_norm_weights, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));

    ggml_build_forward_expand(gf,temp_language_model_head_layer_norm_weights);

    cur = ggml_mul(ctx0, cur, temp_language_model_head_layer_norm_weights);
    cur = ggml_add(ctx0,cur, model.language_model_head_layer_norm_bias);

    ggml_tensor * final_output =   ggml_cont(ctx0,ggml_view_4d(ctx0, cur, 1024, token_count, 4, 1, cur->nb[1], cur->nb[2], cur->nb[3], (1) * sizeof(float) * 1024 )); 

    //std::cout << "mel token count: " << mel_token_count << std::endl;

    ggml_tensor * final_output_2 =   ggml_cont(ctx0,ggml_view_4d(ctx0, cur, 1024, (mel_token_count/4)-2, 4, 1, cur->nb[1], cur->nb[2], cur->nb[3], (token_count) * sizeof(float) * 1024 )); 


    ggml_set_name(final_output_2, "cur");


    ggml_build_forward_expand(gf, final_output_2);
    



    /*
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

    
    ggml_build_forward_expand(ctx0, position);
    ggml_set_name()
    */


    ggml_free(ctx0);
    return gf;

}






/*
 
  ██████╗ ██████╗ ████████╗   ██████╗                       
 ██╔════╝ ██╔══██╗╚══██╔══╝   ╚════██╗                      
 ██║  ███╗██████╔╝   ██║█████╗ █████╔╝                      
 ██║   ██║██╔═══╝    ██║╚════╝██╔═══╝                       
 ╚██████╔╝██║        ██║      ███████╗                      
  ╚═════╝ ╚═╝        ╚═╝      ╚══════╝                      
                                                            
 ███████╗ ██████╗ ██████╗ ██╗    ██╗ █████╗ ██████╗ ██████╗ 
 ██╔════╝██╔═══██╗██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔══██╗
 █████╗  ██║   ██║██████╔╝██║ █╗ ██║███████║██████╔╝██║  ██║
 ██╔══╝  ██║   ██║██╔══██╗██║███╗██║██╔══██║██╔══██╗██║  ██║
 ██║     ╚██████╔╝██║  ██║╚███╔███╔╝██║  ██║██║  ██║██████╔╝
 ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ 
                                                            
 ██████╗  █████╗ ███████╗███████╗                           
 ██╔══██╗██╔══██╗██╔════╝██╔════╝                           
 ██████╔╝███████║███████╗███████╗                           
 ██╔═══╝ ██╔══██║╚════██║╚════██║                           
 ██║     ██║  ██║███████║███████║                           
 ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝                                                       
 
*/
struct ggml_cgraph * autoregressive_graph(
    const autoregressive_model & model,
    struct ggml_allocr * allocr,
    const std::vector<int>  mel_transformer_inputs_vector,
    const std::vector<gpt_vocab::id> & tokens,
    const bool fake_inputs,
    const int n_past,
    const int fixed_position){

    const int token_count = tokens.size();


    static size_t buf_size = ggml_tensor_overhead()*(25 +70*38)  + ggml_graph_overhead();
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

    ggml_tensor * gpt2_input;

    std::cout << "tokens size" << std::endl;
    std::cout << mel_transformer_inputs_vector.size() << std::endl;
    if (fake_inputs) // 4 corresponds to batch of 4 sequences each with length 1
    {

        std::cout << "reached here" << std::endl;

        struct ggml_tensor * text_embedding = ggml_get_rows(ctx0, model.text_embedding_weights,input);
        struct ggml_tensor * text_position_embedding = ggml_get_rows(ctx0, model.text_position_embedding_weights,position);


        struct ggml_tensor * reshaped_latent = ggml_reshape_4d(ctx0, model.conditioning_latent, 1,1,1,1024);

        struct ggml_tensor * embedding = ggml_add(ctx0,text_embedding, text_position_embedding);

        struct ggml_tensor * reshaped_embedding = ggml_reshape_4d(ctx0, embedding, 1,1,token_count,1024);

        struct ggml_tensor * mel_transformer_inputs =   ggml_new_tensor_1d(ctx0, GGML_TYPE_I32,4*mel_transformer_inputs_vector.size());
        ggml_allocr_alloc(allocr, mel_transformer_inputs);
        
        if (!ggml_allocr_is_measure(allocr)) {
            for (int i = 0; i < 4*mel_transformer_inputs_vector.size(); ++i) {
                int v = mel_transformer_inputs_vector[i];
                ggml_backend_tensor_set(mel_transformer_inputs, &v, i*sizeof(int32_t), sizeof(v));
            
            }

        }


        
        mel_transformer_inputs = ggml_reshape_2d(ctx0, mel_transformer_inputs, 4, mel_transformer_inputs_vector.size()); 


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

        mel_embedding = ggml_add(ctx0,mel_embedding, mel_position_embedding);
    
    
        mel_embedding = ggml_reshape_4d(ctx0, mel_embedding, 1, 4, 1, 1024);


        struct ggml_tensor * output = ggml_concat(ctx0, reshaped_latent, reshaped_embedding);

        struct ggml_tensor * repeated_output = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 4 * (tokens.size()+1) * 1024); // todo do this more cleanly, going to rely on 1d copy for same of simplicity
        output = ggml_reshape_1d(ctx0, output, (tokens.size()+1)*1024);


        repeated_output =  ggml_repeat(ctx0, output, repeated_output);
        repeated_output = ggml_reshape_4d(ctx0, repeated_output, 1,4,(tokens.size()+1),1024);



        gpt2_input= ggml_concat(ctx0, repeated_output,mel_embedding);

                
        ggml_tensor * final_latent = ggml_cpy(ctx0, gpt2_input, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,gpt2_input->ne));

        ggml_set_name(final_latent, "gpt2 input");
        ggml_build_forward_expand(gf, final_latent);
 
    }
    else{
        struct ggml_tensor * mel_transformer_inputs =   ggml_new_tensor_1d(ctx0, GGML_TYPE_I32,4); 
        ggml_allocr_alloc(allocr, mel_transformer_inputs);
        
        if (!ggml_allocr_is_measure(allocr)) {
            for (int i = 0; i < 4; ++i) {
                int v = mel_transformer_inputs_vector[i];
                ggml_backend_tensor_set(mel_transformer_inputs, &v, i*sizeof(int32_t), sizeof(v));
            
            }

        }

        
        mel_transformer_inputs = ggml_reshape_2d(ctx0, mel_transformer_inputs, 4, 1); 


        struct ggml_tensor * mel_embedding = ggml_get_rows(ctx0, model.mel_embedding_weights,mel_transformer_inputs);
        ggml_set_name(mel_embedding, "mel embedding");

        struct ggml_tensor * fixed_embedding_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32,1); 
        ggml_allocr_alloc(allocr, fixed_embedding_ids);

        if (!ggml_allocr_is_measure(allocr)) {
            int v = fixed_position;
            ggml_backend_tensor_set(fixed_embedding_ids, &v, 0, sizeof(v));

        }

        ggml_tensor * fixed_embedding = ggml_get_rows(ctx0, model.mel_position_embedding_weights,fixed_embedding_ids);

        ggml_set_name(fixed_embedding, "fixed embedding");
       
       /* if(!fake_inputs && n_past ==19)
        {
                ggml_build_forward_expand(gf, fixed_embedding);
                ggml_set_name(fixed_embedding, "fixedembedding");
                return gf;
        } */

        gpt2_input =  ggml_add(ctx0,mel_embedding, fixed_embedding);
        ggml_set_name(gpt2_input, "gpt2 input");

        //ggml_build_forward_expand(gf, gpt2_input);
        //ggml_free(ctx0);
        //return gf;

    }


     /*
    if ( !fake_inputs && n_past == 19)
    {
                ggml_set_name(gpt2_input, "gpt2 input");
                ggml_build_forward_expand(gf, gpt2_input);
                ggml_free(ctx0);
                return gf;
    }
    */

    int test_dimension = gpt2_input->ne[2];
    std::cout << "test dimension: " << test_dimension << std::endl;
    std::cout << "n_past: " << n_past << std::endl;

    struct ggml_tensor * cur = ggml_reshape_4d(ctx0, gpt2_input, 1024,test_dimension,4,1);


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
    for (int i = 0; i < 30; i++)
    {
           

           residual = ggml_cpy(ctx0, cur, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,1024,test_dimension,4,1));
           
           //ggml_build_forward_expand(gf, residual);
           //layer norm
           
           cur = ggml_norm(ctx0, cur, 1e-05);

           ggml_tensor * temp_cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne) );


           ggml_set_name(temp_cur, "postnorm");
           ggml_build_forward_expand(gf, temp_cur);


           ggml_format_name(cur, "l%d.norm", i);

           ggml_tensor * temp_ln_1_weights = ggml_repeat(ctx0,model.layers[i].linear_1_weights, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));

           ggml_set_name(temp_ln_1_weights, "weights");
           //ggml_build_forward_expand(gf, temp_ln_1_weights);


           cur = ggml_mul(ctx0, cur,temp_ln_1_weights); // if you flip the order of this it doesn't work on the second token generation process.TODO why does flipping the order of this break it?
            
           
           cur = ggml_add(ctx0,cur, model.layers[i].linear_1_bias);
           
          


            //ggml_tensor * temp_weights = ggml_cpy(ctx0, model.layers[i].linear_1_weights, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,model.layers[i].linear_1_weights->ne) );
            ggml_tensor * temp_bias = ggml_cpy(ctx0, model.layers[i].linear_1_bias, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,model.layers[i].linear_1_bias->ne) );
 
          //  if(!fake_inputs)
           // {
            //ggml_build_forward_expand(gf, temp_bias);
            //ggml_set_name(temp_bias, "bias");
           // return gf;
           // } 

                
                
          
            // this is implemented as conv1d in pytorch, but it's actually just a affine transformation with
            // a weight and bias
            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_attention_attention_weights)),1024,3072),
                        cur);


            cur = ggml_reshape_4d(ctx0, cur, 3072,test_dimension,4,1);


            cur = ggml_add(ctx0,cur,
                    model.layers[i].c_attention_attention_bias); // worth studying the diffs here, why did I have to remove this repeat for settings where the second dimension is 1?
            
            
            cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F16,3,cur->ne));
            cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32,3,cur->ne));


         
          


            //derived from ggml reference gpt-2 implementation
            Qcur = ggml_cont(ctx0,ggml_view_3d(ctx0, cur, 1024, test_dimension, 4, cur->nb[1], cur->nb[2], 0));
           

            //Kcur = ggml_cont(ctx0,ggml_permute(ctx0,ggml_view_4d(ctx0, cur, 1024, test_dimension, 4, 1,cur->nb[1], cur->nb[2],cur->nb[3], 1024 * sizeof(float)),0,1,3,2));

            Kcur = ggml_cont(ctx0,ggml_permute(ctx0,ggml_view_3d(ctx0, cur, 1024, test_dimension, 4, cur->nb[1], cur->nb[2], 1024 * sizeof(float)),0,2,1,3));
            Vcur = ggml_cont(ctx0,ggml_permute(ctx0,ggml_view_3d(ctx0, cur, 1024, test_dimension, 4, cur->nb[1], cur->nb[2], 2048 * sizeof(float)),0,2,1,3));

            struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_key, 1024 * test_dimension * 4 , (ggml_element_size(model.memory_key)* ((i * 404 * 1024 * 4) + ((n_past) * 1024 *4))));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
            
            struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_value, 1024 * test_dimension * 4 , (ggml_element_size(model.memory_value)* ((i * 404 * 1024 * 4) + ((n_past) * 1024 *4))));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));


            //num heads 16
            //head dim 64
          
            Q =ggml_cont(ctx0,ggml_permute(ctx0,
                        ggml_reshape_4d(ctx0, Qcur , 64,16,test_dimension,4),
                        0, 2, 1, 3));

            
            // this is likely not general and but should work for the first layer, may need generalizing once we reach
            //the end of the first layer.
            K =ggml_cont(ctx0,ggml_permute(ctx0,
                        ggml_reshape_4d(ctx0, ggml_view_1d(ctx0, model.memory_key, 1024 * (test_dimension + n_past) * 4 , (ggml_element_size(model.memory_key)* (i * 404 * 1024 * 4) )) , 64,16, 4 , test_dimension + n_past),
                        0, 2,3, 1));

          
          

            V_transposed = ggml_cont(ctx0,ggml_permute(ctx0,
                        ggml_reshape_4d(ctx0, ggml_view_1d(ctx0, model.memory_value, 1024 * (test_dimension + n_past) * 4 , (ggml_element_size(model.memory_value)* (i * 404 * 1024 * 4) )) , 64,16,4,test_dimension+n_past),
                        1,2,3,0));

          

            //std::cout << K->ne[0]  << K->ne[1] << K->ne[2]  << K->ne[3] << std::endl;
            //std::cout << Q->ne[0]  << Q->ne[1] << Q->ne[2]  << Q->ne[3] << std::endl;
            //std::cout << V_transposed->ne[0]  << V_transposed->ne[1] << V_transposed->ne[2]  << V_transposed->ne[3] << std::endl;
            KQ = ggml_mul_mat(ctx0, K,Q);

            //std::cout << "KQ shape" << std::endl;
            //std::cout << KQ->ne[0]  << KQ->ne[1] << KQ->ne[2]  << KQ->ne[3] << std::endl;

        
       
          
            

            //KQ = ggml_reshape_1d(ctx0, KQ, KQ->ne[0]* KQ->ne[1]*KQ->ne[2]*KQ->ne[3]);

            //KQ = ggml_cpy(ctx0, KQ, ggml_new_tensor(ctx0, GGML_TYPE_F16,4,KQ->ne));
            //KQ = ggml_cpy(ctx0, KQ, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,KQ->ne));

            
            
            
            KQ_scaled = ggml_scale_inplace(ctx0, KQ, ggml_new_f32(ctx0,1.0f/sqrt(float(64))));


            KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);


            KQ_soft_max =  ggml_soft_max_inplace(ctx0, KQ_masked);
            

            KQV = ggml_mul_mat(ctx0, KQ_soft_max, V_transposed);
            
           //KQV = ggml_mul_mat(ctx0, ggml_reshape_4d(ctx0,ggml_cont(ctx0,ggml_reshape_3d(ctx0, KQ_soft_max, test_dimension + n_past,test_dimension + n_past,64)),test_dimension + n_past,test_dimension+n_past,16,4), ggml_reshape_3d(ctx0,V_transposed,n_past + test_dimension, 16,);
      
            ggml_set_name(KQ_soft_max, "after KQ");
            ggml_set_name(V_transposed, "after V");


          
      
        

           //getting the initial KQV value
           KQV = ggml_reshape_3d(ctx0, KQV, test_dimension ,64,64);
           KQV = ggml_permute(ctx0, KQV, 1,0,2,3);
           KQV = ggml_cont_3d(ctx0, KQV, 64,test_dimension,64);
           KQV = ggml_reshape_4d(ctx0, KQV, 64,test_dimension,16,4);
           

           //"merge heads" operation
           KQV_merged = ggml_permute(ctx0, KQV, 0,2,1,3);
           KQV_merged = ggml_cont_3d(ctx0, KQV_merged, 1024, test_dimension, 4);
    

          
      



            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_attention_projection_weights)),1024,1024),
                        KQV_merged);



            cur = ggml_add(ctx0,cur,
                   model.layers[i].c_attention_projection_bias);




            //layer input passthrough
            //cur = ggml_add(ctx0, cur,residual);

            cur = ggml_add(ctx0, ggml_reshape_1d(ctx0,cur, 1024 * 4 * test_dimension), ggml_reshape_1d(ctx0,residual, 1024 * 4 * test_dimension)); // it's really strange that this is necessary, why does it have different behavior than commented out above 
            cur = ggml_reshape_4d(ctx0, cur, 1024, test_dimension, 4, 1);
    


            




            feed_forward_residual = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));




            //layer norm 2
            cur = ggml_norm(ctx0, cur, 1e-05);

            ggml_format_name(cur, "l%d.norm_2", i);

            ggml_tensor * temp_ln_2_weights = ggml_repeat(ctx0,model.layers[i].linear_2_weights, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));

            ggml_set_name(temp_ln_2_weights, "test");


            ggml_build_forward_expand(gf,temp_ln_2_weights);

            cur = ggml_mul(ctx0, cur, temp_ln_2_weights);
            cur = ggml_add(ctx0,cur, model.layers[i].linear_2_bias);




            //  fully connected multi layer perceptron
            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_multi_layer_perceptron_fully_connected_weights)),1024,4096),
                        cur);



            cur = ggml_add(ctx0,cur, model.layers[i].c_multi_layer_perceptron_fully_connected_bias);

            // gelu
            cur = ggml_gelu(ctx0, cur);


            // mlp fully connected
            cur = ggml_mul_mat(ctx0,
                        ggml_reshape_2d( ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.layers[i].c_multi_layer_perceptron_projection_weights)),4096,1024),
                        cur);


            cur = ggml_add(ctx0, cur, model.layers[i].c_multi_layer_perceptron_projection_bias);



            //final residual
            //another case where I had to flatten before adding to get correct results. Either ggml had a pre-existing bug for batch addition, or one of my modifications introduced it. This will need to be addressed.
            
            cur = ggml_add(ctx0, ggml_reshape_1d(ctx0,cur , 1024 *  test_dimension *  4 *  1) , ggml_reshape_1d(ctx0,feed_forward_residual, 1024 *  test_dimension *  4 *  1));


            cur = ggml_reshape_4d(ctx0, cur, 1024, test_dimension, 4, 1);
            //cur = ggml_add(ctx0, cur, feed_forward_residual);


         
        


    }



    cur = ggml_norm(ctx0, cur, 1e-05);


    ggml_tensor * temp_final_layer_norm_weights = ggml_repeat(ctx0,model.final_layer_norm_weights, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));



    ggml_build_forward_expand(gf,temp_final_layer_norm_weights);



    cur = ggml_mul(ctx0, cur, temp_final_layer_norm_weights);
    cur = ggml_add(ctx0,cur, model.final_layer_norm_bias);


    
    
    cur = ggml_norm(ctx0, cur, 1e-05);



    ggml_tensor * temp_language_model_head_layer_norm_weights = ggml_repeat(ctx0,model.language_model_head_layer_norm_weights, ggml_new_tensor(ctx0, GGML_TYPE_F32,4,cur->ne));


    cur = ggml_mul(ctx0, cur, temp_language_model_head_layer_norm_weights);
    cur = ggml_add(ctx0,cur, model.language_model_head_layer_norm_bias);
    

  

    cur = ggml_mul_mat(ctx0,
                        model.language_model_head_linear_weights,
                        cur);



    cur = ggml_add(ctx0,cur,
             model.language_model_head_linear_bias);

    /*if(!fake_inputs)
    {
            ggml_build_forward_expand(gf, cur);
            ggml_set_name(cur, "afterlmheadlayer");
            return gf;
    } */
        
    
    
    ggml_tensor * next_token_logits = ggml_cont(ctx0,ggml_view_4d(ctx0, cur, 8194, 1, 4, 1, cur->nb[1], cur->nb[2], cur->nb[3], (test_dimension-1) * sizeof(float) * 8194 )); // this "test_dimension - 1" business slices off the last batch of logits

    
    

    
    next_token_logits = ggml_reshape_4d(ctx0, next_token_logits, 8194, 4, 1,1);
   

    ggml_set_name(next_token_logits, "next token logits");


    //mel_transformer_inputs = ggml_reshape_4d(ctx0, mel_transformer_inputs, 18, 4, 1, 1);

    //ggml_tensor * score = ggml_gather(ctx0, next_token_logits, mel_transformer_inputs, 1);
    
    
    std::cout << "didn't reach here" << std::endl;

    ggml_build_forward_expand(gf, next_token_logits);

    //embd_w.resize(n_vocab);
   // memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);



    std::cout << "reached end graph build" << std::endl;

    ggml_free(ctx0);


    return gf;
    
}

/*
 
 ██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗     ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
 ██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗    ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
 ███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝    █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
 ██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗    ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
 ██║  ██║███████╗███████╗██║     ███████╗██║  ██║    ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
 ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝    ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
                                                                                                                               
*/


//great work openchat3.5!!!
void apply_padding(std::vector<int>& vec) {
    // Remove any trailing 8139's from the vector
    while (!vec.empty() && vec.back() == 8139) {
        vec.pop_back();
    }

    // Assert that the vector is shorter than or equal to 500 in length
    assert(vec.size() <= 500);

    // Fill in the int value 83 up to the end of the vector
    for (size_t i = vec.size(); i < 500; ++i) {
        vec.push_back(83);
    }

    // Replace the last 3 ints with 45, 45, 248
    vec[vec.size() - 3] = 45;
    vec[vec.size() - 2] = 45;
    vec[vec.size() - 1] = 248;

    vec.push_back(8193);

    vec.insert(vec.begin(), 8192);
}

template <typename T>
void printVector(std::vector<T> vector, int n, std::string name) {
    std::cout << name << ":\n";
    
    // Print first n elements
    for (int i = 0; i < n && i < vector.size(); i++) {
        std::cout << vector[i] << " ";
    }
    
    std::cout << "\n";
    
    // Print last n elements
    for (int i = vector.size() - n; i < vector.size(); i++) {
        std::cout << vector[i] << " ";
    }
    
    std::cout << std::endl;
}

void printValuesAboveThreshold(const std::vector<float>& vec, float threshold) {
    std::cout << "REACHED 6 " << std::endl;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (vec[i] > threshold) {
            std::cout << "Index: " << i% 8194 << ", Value: " << vec[i] << std::endl;
        }
    }
}

std::vector<float> apply_penalty(const std::vector<float> score, float penalty) {
    std::vector<float> result(score.size());
    for (size_t i = 0; i < score.size(); ++i) {
        result[i] = (score[i] < 0) ? score[i] * penalty : score[i] / penalty;
    }
    return result;
}


std::vector<float> gather(std::vector<float> src, std::vector<int> input_ids)
{

    const int BATCH_SIZE = 4; //hardcoding for now;
    const int sequence_length = input_ids.size()/4;
    const int vocab_size = src.size()/4; //this is 8194, hardcoding for now

    std::vector<float> result(input_ids.size());

    for (int i = 0; i < input_ids.size(); i ++)
    {
    
    const int rowIndex = i / sequence_length;

    const int colIndex = input_ids[i];


    result[i] = src[rowIndex * vocab_size + colIndex];
    }
    std::cout << "gather result" << std::endl;
    return result;
}

std::vector<float> scatter(std::vector<float> src1, std::vector<float> src2,  std::vector<int> input_ids)
{
    std::vector<float> result;
    result.resize(src1.size());
    std::copy(src1.begin(), src1.end(), result.begin());

    const int BATCH_SIZE = 4; //hardcoding for now;
    const int sequence_length = input_ids.size()/4;
    const int vocab_size = src1.size()/4; //this is 8194, hardcoding for now

    //std::vector<float> result(input_ids.size());

    for (int i = 0; i < input_ids.size(); i ++)
    {
    
    const int rowIndex = i / sequence_length;

    const int colIndex = input_ids[i];




    result[rowIndex * vocab_size + colIndex] = src2[i];
    }
    printVector(result, 3, "scatter_result");
    return result;


}

void temp_inplace(std::vector<float> &src, float temp)
{
    for(int i = 0; i < src.size(); i++)
    {
        src[i] /= temp;
    }
}

void val_where_below_thresh(std::vector<float> & src, float threshold, float val)
{
    for (int i = 0; i < src.size(); i++)
    {
        if (src[i] < threshold)
            src[i] = val;
    }
}





float nth_largest(std::vector<float> src, int n)
{
    std::sort(src.begin(), src.end());
    return src[src.size() - n];
}

void top_k_inplace(std::vector<float> & src, int k)
{
    k = std::min(k, 8194);
    float kth_largest_val = nth_largest(src, k);
    val_where_below_thresh(src, kth_largest_val, std::numeric_limits<float>::lowest());
}

void softmax_inplace(std::vector<float> & src)
{
    assert(src.size() == 8194);
    float sum = 0;
    for (int i =0; i < src.size();i++)
    {
            assert(src.size() == 8194);

         src[i] = exp(src[i]);
         sum += src[i];
    }
    for (int j =0; j < src.size();j++)
    {
        assert(src.size() == 8194);
         src[j] /= sum;

    }
}


void top_p_inplace(std::vector<float > & src){

    std::vector<std::pair<float, int>>  src_pairs;
    for(int i = 0; i < src.size(); i++){
        src_pairs.push_back(std::make_pair( src[i], i));
    }


    std::vector<float> sorted_logits;
    std::vector<int> sorted_indices;


    std::sort(src_pairs.begin(), src_pairs.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first;
    });



    for (const auto& pair : src_pairs) {
        sorted_logits.push_back(pair.first);
        sorted_indices.push_back(pair.second);
    }




    // next we perform softmax on the vector of floats sorted_logits
    assert(sorted_logits.size() == 8194);
    softmax_inplace(sorted_logits);
    //next we perform an in place cumulative sum operation on sorted logits

    for(int i = 1; i < sorted_logits.size(); i++)
    {
        sorted_logits[i] += sorted_logits[i-1];
    }


    for (int i = 0; i < src.size()-1; i ++) // this -1 is because for some reason the last token is never zeroed out.
    {
        if (sorted_logits[i] <= 0.2){
        src[src_pairs[i].second]  = std::numeric_limits<float>::lowest();
        }
    }


}


int multinomial( std::vector<float> probs) // worth changing to a binary search at some point, but for now done the simple way
{

    float sample = distribution(generator);
    sample = distribution(generator);

    float cumulative_probability = 0;
    for (int i = 0; i < probs.size(); i++)
    {
        cumulative_probability += probs[i];
        if (cumulative_probability >= sample)
        {
            return i;
        }
    }
    return 8194 - 1; // not supposed to be reached, but defaults to the last probability

}



//takes the raw logits coming out of of a pass through gpt2 and transforms them into a multinomial distribution, then samples from said distribution. 
std::vector<int> process_logits_and_sample(ggml_cgraph * gf, std::vector<int>  &  mel_transformer_inputs_vector, int index)
{



        std::cout << "---------------------------------------------------" << std::endl;
        ggml_tensor * next_token_logits = gf->nodes[gf->n_nodes-1];

        std::cout << "NAME:" << std::endl;
        std::cout << next_token_logits->name << std::endl;
        std::cout << "TYPE" << std::endl;
        std::cout <<  next_token_logits->type << std::endl;
        std::cout << "SHAPE:" << std::endl;
        std::cout << next_token_logits->ne[0]<< std::endl;
        std::cout << next_token_logits->ne[1]<< std::endl;
        std::cout << next_token_logits->ne[2]<< std::endl;
        std::cout << next_token_logits->ne[3]<< std::endl;
        std::cout << "DATA:" << std::endl;

        //save_f32_tensor(next_token_logits, "logs/next_token_logits_" + std::to_string(index) +   ".data");

        int elements = next_token_logits->ne[0] * next_token_logits->ne[1] * next_token_logits->ne[2] * next_token_logits->ne[3];


        std::vector<float> next_token_logits_vector( elements);
        ggml_backend_tensor_get(next_token_logits,next_token_logits_vector.data(), 0 ,sizeof(float)* elements); 
      

        
        std::cout << "reaced end" << std::endl;

        std::vector<float> gather_result =  gather(next_token_logits_vector, mel_transformer_inputs_vector);
        gather_result = apply_penalty(gather_result, 2.0);
        std::cout << "BEGIN" << std::endl;
        std::vector<float> transformed_mel_transformer_inputs_vector = scatter(next_token_logits_vector, gather_result, mel_transformer_inputs_vector);
        
        std::cout << transformed_mel_transformer_inputs_vector.size() << std::endl;
        std::vector<int> samples(4); // batch size is 4


        std::vector<float> probs(transformed_mel_transformer_inputs_vector.size());
        for(int i = 0; i < 4; i ++) // hardcoded to batch size of 4
        {

            std::vector<float> logits;
            logits.insert(logits.end(), transformed_mel_transformer_inputs_vector.begin() + (i * 8194), transformed_mel_transformer_inputs_vector.begin() + ((i+1)*8194));
            assert(logits.size() == 8194);
            //transformed_mel_transformer_inputs_vector.resize(8194); // just considering 1 out of 4 in the batch for now for testing purposes;
            temp_inplace(logits, 0.8);
            top_k_inplace(logits, 50);
            top_p_inplace(logits);

            softmax_inplace(logits); // they become probs at this point, so name is misleading now
            samples[i] = multinomial(logits);

            for (int c = 0; c <  8194; c++){
                probs[i*8194 + c] = logits[c];
            }

        }
        //probs
        printValuesAboveThreshold(probs, .01);
        return samples;
}





//trims latents to no more than 8 calm tokens at the end
//the latents are each 1024 x 500, and there is a number equal to the batch size. 
std::vector<std::vector<float>> trim_latents(std::vector<float> & latents, std::vector<std::vector<int>>  & mel_codes)
{
    std::cout << mel_codes.size(); 

    int total_sequence_length = 0;
    for (int i =0; i < mel_codes.size(); i ++)
    {

    // Remove first element
        mel_codes[i].erase(mel_codes[i].begin());
        
        // Remove last element
        mel_codes[i].erase(mel_codes[i].end()-1);

        std::cout << mel_codes[i].size() << std::endl;
        total_sequence_length += mel_codes[i].size();
        std::cout << "size" << std::endl;
        std::cout << mel_codes[i].size() << std::endl;
        localAssert (mel_codes[i].size() == 500);
        
    }

    std::cout <<"reached" << std::endl;
    std::cout << total_sequence_length << std::endl;
    std::cout << latents.size() << std::endl;
    // localAssert (total_sequence_length == latents.size());


    std::vector<std::vector<float>> trimmed_latents (mel_codes.size());


    for (int i =0 ; i < mel_codes.size(); i ++)
    {
        const int offset = i * 500*1024;
        int calm_tokens = 0;
        
        
        for (int c = 0 ; c < 500; c++)
        {

            if (mel_codes[i][c] == 83)
            {
                calm_tokens += 1;
            }
            else{
                calm_tokens = 0;
            }
            if (calm_tokens > 8)
            {
                break;
            }
            for (int j = 0; j < 1024; j++)
            {
            trimmed_latents[i].push_back(latents[offset + (c*1024) + j]);
            }
            
        }
        std::cout << " " << trimmed_latents[i].size() << " ";
    }
    return trimmed_latents;
}


// prints either all leaves from a computational graph, or all the nodes
void print_all_tensors(struct ggml_cgraph * gf, bool leaves, bool filter_flag, std::string filter){

    int count = leaves ? gf->n_leafs : gf->n_nodes;

    for (int i =0; i < count; i ++)
    {
        ggml_tensor * test;
        if (leaves){
         test = gf->leafs[i];
        }
        else{
         test = gf->nodes[i];
        }
        
        
        
         if (filter_flag && std::string(test->name) != filter )
        {
            continue;
        }

        save_f32_tensor(test, "logs/" + std::string(test->name) + ".txt");

        std::cout << "---------------------------------------------------" << std::endl;
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
                
            if  (c < 3 || c > elements-4  || c == 2*1024 || c == 1024)
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
        else if(test->type == GGML_TYPE_I32){
        std::vector<int32_t> test_read( elements);
        ggml_backend_tensor_get(test,test_read.data(), 0 ,sizeof(int32_t)* elements);
    //        
        for (int c = 0; c < elements ; c++)
        {
            if  (c < 3 || c > elements-4)
            {
            
            std::cout << test_read.data()[c]<< std::endl;
            }
        }
        }
        



    }

}



/*
 
 ███╗   ███╗ █████╗ ██╗███╗   ██╗
 ████╗ ████║██╔══██╗██║████╗  ██║
 ██╔████╔██║███████║██║██╔██╗ ██║
 ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
 ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
 ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
                                 
 
*/

int main(int argc, char ** argv) {

    std::cout << "hello world" << std::endl;
    

    
    //std::uniform_real_distribution<float> distribution(0.0, 1.0);
    /*
    std::cout << distribution(generator) << std::endl;
    std::cout << distribution(generator) << std::endl;
    std::cout << distribution(generator) << std::endl;
    std::cout << distribution(generator) << std::endl;
        std::cout << distribution(generator) << std::endl;
    std::cout << distribution(generator) << std::endl;
    std::cout << distribution(generator) << std::endl;
    std::cout << distribution(generator) << std::endl;
    */
    gpt_vocab vocab;
    gpt_vocab_init("../examples/tortoise/tokenizer.json", vocab);
    
    std::string message = "this[SPACE]is[SPACE]a[SPACE]test[SPACE]message";
    //std::vector<gpt_vocab::id> tokens = ::gpt_tokenize(vocab, message);
    //std::vector<gpt_vocab::id> tokens = ::parse_tokens_from_string("255,147,2,54,2,14,2,33,218,2,26,61,150,112,0,0", ','); // for now, skipping some token processing steps
    //std::vector<gpt_vocab::id> tokens = ::parse_tokens_from_string("255,147,2,54,2,14,2,33,218,2,26,61,150,112,0,0", ','); // "This is a test message"
    //std::vector<gpt_vocab::id> tokens = ::parse_tokens_from_string("255,15,55,49,9,9,9,2,134,16,51,31,2,19,46,18,176,13,0,0", ','); //"Based... Dr. Freeman?"
    //std::vector<gpt_vocab::id> tokens = ::parse_tokens_from_string("255,135,198,48,167,158,32,3,2,14,34,51,46,20,175,212,76,2,115,126,25,2,170,29,64,136,3,0,0,255,135,198,48,167,158,32,3,2,14,34,51,46,20,175,212,76,2,115,126,25,2,170,29,64,136,3,0,0", ','); //"Congratulations! Autoregressive model complete!"
    //exit(0);
    std::vector<gpt_vocab::id> tokens = ::parse_tokens_from_string("255,42,2,97,60,49,2,63,48,61,2,26,163,2,149,2,68,161,33,2,42,2,33,77,78,16,32,2,58,2,42,2,50,18,125,9,2,80,43,32,2,127,2,106,29,57,33,159,7,2,55,2,204,32,9,2,16,31,54,54,2,26,213,61,2,60,2,136,26,29,242,2,51,2,22,20,95,46,2,42,2,36,54,18,2,46,63,31,137,192,2,73,2,26,245,2,26,50,2,19,46,18,9,2,0,0", ','); // "The United States must not adopt the tactics of the enemy. Means are important, as ends. Crisis makes it tempting to ignore the wise restraints that make men free." - Frank Church



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


    std::vector<int> mel_transformer_inputs_vector = std::vector<int>();
    mel_transformer_inputs_vector.resize((tokens.size() + 2) * 4);
    //assert(tokens.size() == 16);
    
    for (int i = 0; i < mel_transformer_inputs_vector.size(); i ++)
    {
        if (i % (tokens.size()+2) == tokens.size()+2-1){
            mel_transformer_inputs_vector[i] = 8192;
        }
        else{
            mel_transformer_inputs_vector[i] = 1;
        }
    }



    ggml_backend_buffer_t buf_compute;

    struct ggml_allocr * allocr = NULL;
    // allocate the compute buffer

    // alignment required by the backend
    size_t align = ggml_backend_get_alignment(model.backend);
    std::cout << "alignment" << std::endl;
    std::cout << align << std::endl;
    allocr = ggml_allocr_new_measure(align);
    std::cout << "align created" << std::endl;

    // create the worst case graph for memory usage estimation
    //int n_tokens = std::min(model.hparams.n_ctx, params.n_batch);
    //int n_past = model.hparams.n_ctx - n_tokens;
    ggml_allocr_reset(allocr);
    struct ggml_cgraph * gf = autoregressive_graph(model, allocr, mel_transformer_inputs_vector, tokens, true, 0,0);
    //ggml_graph_print(gf);

    std::cout << "graph created" << std::endl;
    // compute the required memory
    size_t mem_size = ggml_allocr_alloc_graph(allocr, gf);

    // recreate the allocator with the required memory
    ggml_allocr_reset(allocr);
    buf_compute = ggml_backend_alloc_buffer(model.backend, mem_size);
    allocr = ggml_allocr_new_from_buffer(buf_compute);
    gf = autoregressive_graph(model, allocr,mel_transformer_inputs_vector, tokens, true, 0,0);
    ggml_allocr_alloc_graph(allocr, gf);
    std::cout << "reached computing time" << std::endl;
    ggml_backend_graph_compute(model.backend, gf);
    //ggml_graph_print(gf);
    std::vector<int> samples;
    
    std::string sample_string; 
    int stop_token = 8193;
    bool all_sequences_stopped = false;

    std::vector<std::vector<int>> sequences(4);

    int i = 0;
    while (!all_sequences_stopped)
    {
    samples =  process_logits_and_sample(gf,  mel_transformer_inputs_vector, i);
    
    
    printVector(samples, 2, "samples");

    sample_string = sample_string + ",[";

    int stop_token_count = 0;

    mel_transformer_inputs_vector.clear();
    for (int c = 0; c < 4; c ++)
    {
        if (!(sequences[c].size()>0 && sequences[c][sequences[c].size()-1] == stop_token))
        {
            sequences[c].push_back(samples[c]);
        }
        if (samples[c] ==stop_token)
        {
            stop_token_count += 1;
        }
        mel_transformer_inputs_vector.push_back(samples[c]);
        sample_string = sample_string  + std::to_string(samples[c]) + ',';
    }
    if (stop_token_count == 4)
    {
        all_sequences_stopped = true;
    }
    sample_string = sample_string + "]";

    ggml_allocr_reset(allocr);
    allocr = ggml_allocr_new_from_buffer(buf_compute);
    gf = autoregressive_graph(model, allocr,mel_transformer_inputs_vector, tokens, false, tokens.size() + 2 + i, i+2);
    ggml_allocr_alloc_graph(allocr, gf);
    std::cout << "reached computing time" << std::endl;
    ggml_backend_graph_compute(model.backend, gf);
    i+= 1;
    }
    ggml_allocr_free(allocr);

    print_all_tensors(gf, false, true, "gpt2 input");


    for (int i = 0; i < 4; i ++)
    {
        apply_padding(sequences[i]);
    }


    // Iterate through the outer vector
    for (std::vector<int> inner_vector : sequences) {
        // Print the inner vector as a Python list literal
        std::cout << "[";
        for (size_t i = 0; i < inner_vector.size(); ++i) {
            std::cout << inner_vector[i];
            if (i < inner_vector.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }



    mel_transformer_inputs_vector.clear();
    for (int c = 0; c < 4; c ++)
    {
        for (int i = 0; i < 502; i ++)
        {
            mel_transformer_inputs_vector.push_back(sequences[c][i]);
        }
    }


    /*
    ggml_allocr_reset(allocr);
    buf_compute = ggml_backend_alloc_buffer(model.backend, mem_size);
    allocr = ggml_allocr_new_from_buffer(buf_compute);
    gf = autoregressive_graph(model, allocr,mel_transformer_inputs_vector, tokens, true, 0,0);
    ggml_allocr_alloc_graph(allocr, gf);
    std::cout << "reached computing time" << std::endl;
    ggml_backend_graph_compute(model.backend, gf);
    */



    ggml_backend_buffer_t latent_buf_compute;

    struct ggml_allocr * latent_allocr = NULL;
    // allocate the compute buffer

    // alignment required by the backend
    size_t latent_align = ggml_backend_get_alignment(model.backend);
    std::cout << "alignment" << std::endl;
    std::cout << latent_align << std::endl;
    latent_allocr = ggml_allocr_new_measure(latent_align);
    std::cout << "align created" << std::endl;

    // create the worst case graph for memory usage estimation
    //int n_tokens = std::min(model.hparams.n_ctx, params.n_batch);
    //int n_past = model.hparams.n_ctx - n_tokens;
    ggml_allocr_reset(latent_allocr);
    struct ggml_cgraph * latent_gf = autoregressive_latent_graph(model, latent_allocr, mel_transformer_inputs_vector, tokens);
    //ggml_graph_print(gf);

    std::cout << "graph created" << std::endl;
    // compute the required memory
    size_t latent_mem_size = ggml_allocr_alloc_graph(latent_allocr, latent_gf);



    ggml_allocr_reset(latent_allocr);
    latent_buf_compute = ggml_backend_alloc_buffer(model.backend, latent_mem_size);
    latent_allocr = ggml_allocr_new_from_buffer(latent_buf_compute);
    latent_gf = autoregressive_latent_graph(model, latent_allocr,mel_transformer_inputs_vector, tokens);
    ggml_allocr_alloc_graph(latent_allocr, latent_gf);
    std::cout << "reached computing time" << std::endl;
    ggml_backend_graph_compute(model.backend, latent_gf);


    std::cout << "Produced autoregressive latents :^)" << std::endl;
    //print_all_tensors(latent_gf, true, true, "cur");
    print_all_tensors(latent_gf, false, true, "cur");

    std::vector<float> latents = std::vector<float>();

    extract_latents_to_vector(  latent_gf->nodes[latent_gf->n_nodes -1] , latents);

    

    ggml_allocr_free(latent_allocr);
    ggml_allocr_free(allocr);


    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer_w);
    ggml_backend_buffer_free(buf_compute);
    ggml_backend_buffer_free(latent_buf_compute);

    ggml_backend_free(model.backend);


    std::vector<std::vector<float>> trimmed_latents = trim_latents(latents, sequences);

    std::cout <<"reached2" << std::endl;
    for (int i = 0; i < trimmed_latents[0].size(); i ++)
    {
        if (i < 3 || i > trimmed_latents[0].size() -4)
        {
        std::cout << "trimmed latents 1\n" << trimmed_latents[0][i] << std::endl;
        }
    }
    std::cout << "length: " << trimmed_latents[0].size();

    /*
    std::cout << "sequences" << std::endl;

    // Iterate through the outer vector
    for (std::vector<int> inner_vector : sequences) {
        // Print the inner vector as a Python list literal
        std::cout << "[";
        for (size_t i = 0; i < inner_vector.size(); ++i) {
            std::cout << inner_vector[i];
            if (i < inner_vector.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
    */
        
    return 0;
  
}
