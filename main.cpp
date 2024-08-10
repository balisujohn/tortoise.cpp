#include "ggml/include/ggml-alloc.h"
#include "ggml/include/ggml-backend.h"
#include "ggml/include/ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "common.h"

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

#define GPT2_MAX_NODES 4096
#define TORTOISE_NORM_EPSILON 1e-8

int32_t NUM_RETURN_SEQUENCES =
    4; // hardcoding this for now, analagous to "num_return_sequences arugment
       // to inference_speech"

auto now = std::chrono::system_clock::now();
auto duration = now.time_since_epoch();
auto milliseconds =
    std::chrono::duration_cast<std::chrono::milliseconds>(duration);

// Use the milliseconds count as a seed for the random number generator
unsigned time_seed = milliseconds.count();

std::mt19937 generator(time_seed);

std::uniform_real_distribution<float> distribution(0.0, 1.0);
std::normal_distribution<double> normal_distribution(0.0, 1.0);

void localAssert(bool condition) {
  if (!condition) {
    std::cout << "failure" << std::endl;
    exit(0);
  }
}

// clang-format off

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

// clang-format on

// derived from ggml gpt2 reference implementation
struct gpt2_layer {
  struct ggml_tensor *layer_norm_1_weights;
  struct ggml_tensor *layer_norm_1_bias;

  struct ggml_tensor *layer_norm_2_weights;
  struct ggml_tensor *layer_norm_2_bias;

  // attention
  struct ggml_tensor *c_attention_attention_weights;
  struct ggml_tensor *c_attention_attention_bias;

  struct ggml_tensor *c_attention_projection_weights;
  struct ggml_tensor *c_attention_projection_bias;

  // mlp
  struct ggml_tensor *c_multi_layer_perceptron_fully_connected_weights;
  struct ggml_tensor *c_multi_layer_perceptron_fully_connected_bias;

  struct ggml_tensor *c_multi_layer_perceptron_projection_weights;
  struct ggml_tensor *c_multi_layer_perceptron_projection_bias;
};

struct autoregressive_model {

  struct ggml_tensor *embedding;

  std::map<std::string, struct ggml_tensor *> tensors;

  struct ggml_tensor *text_embedding_weights;
  struct ggml_tensor *text_position_embedding_weights;

  struct ggml_tensor *mel_embedding_weights;
  struct ggml_tensor *mel_position_embedding_weights;

  struct ggml_tensor *final_layer_norm_weights;
  struct ggml_tensor *final_layer_norm_bias;

  struct ggml_tensor *language_model_head_layer_norm_weights;
  struct ggml_tensor *language_model_head_layer_norm_bias;

  struct ggml_tensor *language_model_head_linear_weights;
  struct ggml_tensor *language_model_head_linear_bias;

  struct ggml_tensor *memory_key;
  struct ggml_tensor *memory_value;

  std::vector<gpt2_layer> layers;

  struct ggml_context *ctx;

  ggml_backend_buffer_t buffer_w;

  ggml_backend_t backend = NULL;
};

// clang-format off

/*

 ██████╗ ██╗███████╗███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗
 ██╔══██╗██║██╔════╝██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║
 ██║  ██║██║█████╗  █████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║
 ██║  ██║██║██╔══╝  ██╔══╝  ██║   ██║╚════██║██║██║   ██║██║╚██╗██║
 ██████╔╝██║██║     ██║     ╚██████╔╝███████║██║╚██████╔╝██║ ╚████║
 ╚═════╝ ╚═╝╚═╝     ╚═╝      ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝

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

// clang-format on

struct latent_conditioner_attention_block {

  struct ggml_tensor *norm_weight;
  struct ggml_tensor *norm_bias;

  struct ggml_tensor *qkv_weight;
  struct ggml_tensor *qkv_bias;

  struct ggml_tensor *projection_out_weight;
  struct ggml_tensor *projection_out_bias;

  struct ggml_tensor
      *relative_position_embeddings_relative_attention_bias_weight;
};

struct residual_block {

  struct ggml_tensor *in_layers_0_weight;

  struct ggml_tensor *in_layers_0_bias;

  struct ggml_tensor *in_layers_2_weight;

  struct ggml_tensor *in_layers_2_bias;

  struct ggml_tensor *emb_layers_1_weight;

  struct ggml_tensor *emb_layers_1_bias;

  struct ggml_tensor *out_layers_0_weight;

  struct ggml_tensor *out_layers_0_bias;

  struct ggml_tensor *out_layers_3_weight;

  struct ggml_tensor *out_layers_3_bias;
};

struct diffusion_layer {

  struct ggml_tensor *resblock_in_layers_0_weight;

  struct ggml_tensor *resblock_in_layers_0_bias;

  struct ggml_tensor *resblock_in_layers_2_weight;

  struct ggml_tensor *resblock_in_layers_2_bias;

  struct ggml_tensor *resblock_emb_layers_1_weight;

  struct ggml_tensor *resblock_emb_layers_1_bias;

  struct ggml_tensor *resblock_out_layers_0_weight;

  struct ggml_tensor *resblock_out_layers_0_bias;

  struct ggml_tensor *resblock_out_layers_3_weight;

  struct ggml_tensor *resblock_out_layers_3_bias;

  struct ggml_tensor *attn_norm_weight;

  struct ggml_tensor *attn_norm_bias;

  struct ggml_tensor *attn_qkv_weight;

  struct ggml_tensor *attn_qkv_bias;

  struct ggml_tensor *attn_proj_out_weight;

  struct ggml_tensor *attn_proj_out_bias;

  struct ggml_tensor
      *attn_relative_pos_embeddings_relative_attention_bias_weight;
};

struct diffusion_model {

  struct ggml_tensor *diffusion_conditioning_latent;

  struct ggml_tensor *latent_conditioner_convolution_weight;

  struct ggml_tensor *latent_conditioner_convolution_bias;

  std::vector<latent_conditioner_attention_block>
      latent_conditioner_attention_blocks;

  struct ggml_tensor *code_norm_weight;

  struct ggml_tensor *code_norm_bias;

  struct ggml_tensor *time_emb_linear_0_weight;

  struct ggml_tensor *time_emb_linear_0_bias;

  struct ggml_tensor *time_emb_linear_1_weight;

  struct ggml_tensor *time_emb_layer_norm_1_bias;

  std::vector<diffusion_layer>
      timestep_conditioning_integrator_diffusion_layers;

  struct ggml_tensor *inp_block_weight;

  struct ggml_tensor *inp_block_bias;

  struct ggml_tensor *integrating_conv_weight;

  struct ggml_tensor *integrating_conv_bias;

  std::vector<diffusion_layer> main_diffusion_layers;

  std::vector<residual_block> main_residual_blocks;

  struct ggml_tensor *out_group_norm_weight;
  struct ggml_tensor *out_group_norm_bias;

  struct ggml_tensor *out_convolution_weight;
  struct ggml_tensor *out_convolution_bias;

  struct ggml_tensor *unconditioned_embedding;

  std::map<std::string, struct ggml_tensor *> tensors;

  struct ggml_context *ctx;

  ggml_backend_buffer_t buffer_w;

  ggml_backend_t backend = NULL;
};

// clang-format off

/*                                        
                                                                  
 ██╗   ██╗ ██████╗  ██████╗ ██████╗ ██████╗ ███████╗██████╗       
 ██║   ██║██╔═══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗      
 ██║   ██║██║   ██║██║     ██║   ██║██║  ██║█████╗  ██████╔╝      
 ╚██╗ ██╔╝██║   ██║██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗      
  ╚████╔╝ ╚██████╔╝╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║      
   ╚═══╝   ╚═════╝  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝      
                                                                  
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
// clang-format on

struct residual_conv_block {
  struct ggml_tensor *residual_convs_1_bias;
  struct ggml_tensor *residual_convs_1_weight;

  struct ggml_tensor *residual_convs_3_bias;
  struct ggml_tensor *residual_convs_3_weight;
};

struct conv_block {
  struct ggml_tensor *conv_block_1_bias;
  struct ggml_tensor *conv_block_1_weight;
};

struct vocoder_residual_block {

  struct ggml_tensor *kernel_predictor_input_convolution_weight;
  struct ggml_tensor *kernel_predictor_input_convolution_bias;

  std::vector<residual_conv_block> kernel_predictor_residual_conv_blocks;

  struct ggml_tensor *kernel_predictor_kernel_convolution_weight;
  struct ggml_tensor *kernel_predictor_kernel_convolution_bias;

  struct ggml_tensor *kernel_predictor_bias_convolution_weight;
  struct ggml_tensor *kernel_predictor_bias_convolution_bias;

  struct ggml_tensor *convolution_t_pre_weight;
  struct ggml_tensor *convolution_t_pre_bias;

  std::vector<conv_block> conv_blocks;
};

// model tether
struct vocoder_model {

  struct ggml_tensor *convolution_pre_weight;
  struct ggml_tensor *convolution_pre_bias;

  std::vector<vocoder_residual_block> residual_stack;

  struct ggml_tensor *convolution_post_weight;
  struct ggml_tensor *convolution_post_bias;

  std::map<std::string, struct ggml_tensor *> tensors;

  struct ggml_context *ctx;

  ggml_backend_buffer_t buffer_w;

  ggml_backend_t backend = NULL;
};

void save_f32_tensor(ggml_tensor *tensor, std::string path_name) {
  std::ofstream stream;
  stream.open(path_name, std::ios::out | std::ios::binary);

  int elements = tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];

  std::vector<float> data_read(elements);
  ggml_backend_tensor_get(tensor, data_read.data(), 0,
                          sizeof(float) * elements);
  stream.write(reinterpret_cast<const char *>(data_read.data()),
               elements * sizeof(float));
  stream.close();
}

void compare_to_saved_tensor_with_name(ggml_tensor *tensor) {

  std::string filename = "./logs/" + std::string(tensor->name) + ".txt";

  int nBytes = tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3] *
               sizeof(float);

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
    return;
  }

  // Calculate number of floats to read based on number of bytes
  size_t numFloats = nBytes / sizeof(float);
  std::vector<float> floats(numFloats);

  // Read floats from file
  file.read(reinterpret_cast<char *>(floats.data()), nBytes);

  file.close();

  std::vector<float> tensor_floats(numFloats);
  ggml_backend_tensor_get(tensor, tensor_floats.data(), 0, nBytes);

  std::cout << "starting comparison" << std::endl;

  bool write_flag = true;
  int lastIndex = -1;
  for (int i = 0; i < floats.size(); i++) {
    // std::cout << floats[i] << std::endl;
    if (abs(floats[i] - tensor_floats[i]) > .01) {
      if (write_flag) {
        std::cout << i << ": " << floats[i] << "," << tensor_floats[i]
                  << std::endl;
      }
      lastIndex = i;
      write_flag = false;
    }
  }
  if (lastIndex != -1) {
    std::cout << "last index " << lastIndex << ": " << floats[lastIndex] << ","
              << tensor_floats[lastIndex] << std::endl;
  }

  std::cout << "done with comparison" << std::endl;
}

void extract_tensor_to_vector(ggml_tensor *tensor, std::vector<float> &vector) {
  int elements = tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
  vector.resize(elements);
  ggml_backend_tensor_get(tensor, vector.data(), 0, sizeof(float) * elements);
}

// clang-format off

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
// clang-format on

// derived from  gpt2_model_load(const std::string & fname, gpt2_model & model,
// gpt_vocab & vocab, int n_ctx, int n_gpu_layers) {
bool autoregressive_model_load(const std::string &fname,
                               autoregressive_model &model) {
  printf("%s: loading model from '%s'\n", __func__, fname.c_str());

  auto fin = std::ifstream(fname, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    return false;
  }

  // verify magic
  {
    uint32_t magic;
    fin.read((char *)&magic, sizeof(magic));
    if (magic != GGML_FILE_MAGIC) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__,
              fname.c_str());
      return false;
    }
  }

  // load hparams
  /*
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

      fin.read((char *) &hparams.max_mel_tokens,
  sizeof(hparams.max_mel_tokens)); fin.read((char *) &hparams.max_text_tokens,
  sizeof(hparams.max_text_tokens)); fin.read((char *)
  &hparams.max_conditioning_inputs, sizeof(hparams.max_conditioning_inputs));
      fin.read((char *) &hparams.layers, sizeof(hparams.layers));
      fin.read((char *) &hparams.model_dim, sizeof(hparams.model_dim));
      fin.read((char *) &hparams.heads, sizeof(hparams.heads));
      fin.read((char *) &hparams.number_text_tokens,
  sizeof(hparams.number_text_tokens)); fin.read((char *)
  &hparams.start_text_token, sizeof(hparams.start_text_token)); fin.read((char
  *) &hparams.num_embeddings, sizeof(hparams.num_embeddings));

      printf("%s: max_mel_tokens = %d\n", __func__, hparams.max_mel_tokens);
      printf("%s: max_text_tokens = %d\n", __func__, hparams.max_text_tokens);
      printf("%s: max_conditioning_inputs =  %d\n", __func__,
  hparams.max_conditioning_inputs); printf("%s: layers = %d\n", __func__,
  hparams.layers); printf("%s: model_dim = %d\n", __func__, hparams.model_dim);
      printf("%s: heads = %d\n", __func__, hparams.heads);
      printf("%s: number_text_tokens =  %d\n", __func__,
  hparams.number_text_tokens); printf("%s: start_text_token =  %d\n", __func__,
  hparams.start_text_token); printf("%s: num_embeddings =  %d\n", __func__,
  hparams.num_embeddings);


  }
  */

  size_t buffer_size = 0;

  buffer_size += ggml_row_size(GGML_TYPE_F32, 256 * 1024); // text embedding weights

  buffer_size += ggml_row_size(GGML_TYPE_F32, 404 * 1024); // text position embedding weights

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1 * 1024); // conditioning latent

  buffer_size += ggml_row_size(GGML_TYPE_F32, 8194 * 1024); // mel embedding weight

  buffer_size += ggml_row_size(GGML_TYPE_F32, 608 * 1024); // mel position embedding weight

  for (int i = 0; i < 30; i++) {
    // todo fix this
    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // inference model linear 1 weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // inference model linear 1 bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 3072); // inference model attention weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 3072); // inference model attention bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024); // inference model attention projection weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // inference model attention projection bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // inference model linear 2 weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // inference model linear 2 bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 4096); // inference model multi layer
                                        // perceptron fully connected weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 4096); // inference model multi layer
                                              // perceptron fully connected bais

    buffer_size += ggml_row_size(GGML_TYPE_F32, 4096 * 1024); // inference model multi layer
                                        // perceptron projection weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // inference model multi layer
                                               // perceptron projection bais
  }

  buffer_size += ggml_row_size(GGML_TYPE_F32, 404 * 30 * 1024 * 4); // key cache (memory_value)
  buffer_size += ggml_row_size(GGML_TYPE_F32, 404 * 30 * 1024 * 4); // value cache (memory_value)

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // final layer norm weight
  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // final layer norm bias

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // language model head layer norm weight
  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // language model head layer norm bias

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 8194); // language model head linear weight
  buffer_size += ggml_row_size(GGML_TYPE_F32, 8194); // language model head linear bias

  printf("%s: ggml tensor size    = %d bytes\n", __func__,
         (int)sizeof(ggml_tensor));
  printf("%s: backend buffer size = %6.2f MB\n", __func__,
         buffer_size / (1024.0 * 1024.0));

  struct ggml_init_params params = {
      /*.mem_size   =*/ggml_tensor_overhead() * (size_t)(5 + 12 * 30 + 8),
      /*.mem_buffer =*/NULL,
      /*.no_alloc   =*/true,
  };

  model.ctx = ggml_init(params);

  if (!model.ctx) {
    fprintf(stderr, "%s: ggml_init() failed\n", __func__);
    return false;
  }

  // initialize the backend
#ifdef GGML_USE_CUDA
  fprintf(stderr, "%s: using CUDA backend\n", __func__);
  model.backend = ggml_backend_cuda_init(0);
  if (!model.backend) {
    fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
  }
#endif

#ifdef GGML_USE_METAL
  fprintf(stderr, "%s: using Metal backend\n", __func__);
  // ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
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

  // model.buffer_w = ggml_backend_alloc_buffer(model.backend, buffer_size);

  auto &ctx = model.ctx;

  model.text_embedding_weights =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 256);
  model.text_position_embedding_weights =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 404);
  model.mel_embedding_weights =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 8194);
  model.mel_position_embedding_weights =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 608);
  model.final_layer_norm_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
  model.final_layer_norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
  model.language_model_head_layer_norm_weights =
      ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
  model.language_model_head_layer_norm_bias =
      ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
  model.language_model_head_linear_weights =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 8194);
  model.language_model_head_linear_bias =
      ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8194);

  model.layers.resize(30);
  for (int i = 0; i < 30; i++) {
    auto &layer = model.layers[i];

    layer.layer_norm_1_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.layer_norm_1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

    layer.c_attention_attention_weights =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3072, 1024);
    layer.c_attention_attention_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3072);

    layer.c_attention_projection_weights =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
    layer.c_attention_projection_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

    layer.layer_norm_2_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.layer_norm_2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

    layer.c_multi_layer_perceptron_fully_connected_weights =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1024);
    layer.c_multi_layer_perceptron_fully_connected_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);

    layer.c_multi_layer_perceptron_projection_weights =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 4096);
    layer.c_multi_layer_perceptron_projection_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".ln_1.weight"] = layer.layer_norm_1_weights;
    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".ln_1.bias"] = layer.layer_norm_1_bias;

    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".attn.c_attn.weight"] = layer.c_attention_attention_weights;
    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".attn.c_attn.bias"] = layer.c_attention_attention_bias;

    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".attn.c_proj.weight"] = layer.c_attention_projection_weights;
    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".attn.c_proj.bias"] = layer.c_attention_projection_bias;

    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".ln_2.weight"] = layer.layer_norm_2_weights;
    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".ln_2.bias"] = layer.layer_norm_2_bias;

    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".mlp.c_fc.weight"] =
        layer.c_multi_layer_perceptron_fully_connected_weights;
    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".mlp.c_fc.bias"] =
        layer.c_multi_layer_perceptron_fully_connected_bias;

    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".mlp.c_proj.weight"] =
        layer.c_multi_layer_perceptron_projection_weights;
    model.tensors["inference_model.transformer.h." + std::to_string(i) +
                  ".mlp.c_proj.bias"] =
        layer.c_multi_layer_perceptron_projection_bias;
  }

  // model.init_conv_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1,1024);

  // model.init_conv_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 80,1024);

  // model.tensors["conditioning_encoder.init.bias"] = model.init_conv_bias;
  // model.tensors["conditioning_encoder.init.weight"] =
  // model.init_conv_weights;
  model.tensors["inference_model.lm_head.0.weight"] =
      model.language_model_head_layer_norm_weights;
  model.tensors["inference_model.lm_head.0.bias"] =
      model.language_model_head_layer_norm_bias;

  model.tensors["inference_model.lm_head.1.weight"] =
      model.language_model_head_linear_weights;
  model.tensors["inference_model.lm_head.1.bias"] =
      model.language_model_head_linear_bias;

  model.tensors["inference_model.transformer.ln_f.weight"] =
      model.final_layer_norm_weights;
  model.tensors["inference_model.transformer.ln_f.bias"] =
      model.final_layer_norm_bias;
  model.tensors["text_embedding.weight"] = model.text_embedding_weights;
  model.tensors["text_pos_embedding.emb.weight"] =
      model.text_position_embedding_weights;
  model.tensors["mel_embedding.weight"] = model.mel_embedding_weights;
  model.tensors["mel_pos_embedding.emb.weight"] =
      model.mel_position_embedding_weights;

  model.memory_key =
      ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * 404 * 30 * 1024);
  model.memory_value =
      ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * 404 * 30 * 1024);

  // ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer_w);

  model.buffer_w = ggml_backend_alloc_ctx_tensors(ctx, model.backend);

  // load weights
  {
    size_t total_size = 0;

    bool has_lm_head = false;

    std::vector<char> read_buf;

    while (true) {
      int32_t n_dims;
      int32_t length;
      int32_t ttype;

      fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
      fin.read(reinterpret_cast<char *>(&length), sizeof(length));
      fin.read(reinterpret_cast<char *>(&ttype), sizeof(ttype));

      if (fin.eof()) {
        break;
      }

      int32_t nelements = 1;
      int32_t ne[4] = {1, 1, 1, 1};
      for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
      }

      std::string name(length, 0);
      fin.read(&name[0], length);

      if (model.tensors.find(name) == model.tensors.end()) {
        fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__,
                name.c_str());
        return false;
      }

      auto tensor = model.tensors[name];
      ggml_set_name(tensor, name.c_str());
      if (ggml_nelements(tensor) != nelements) {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n",
                __func__, name.c_str());
        return false;
      }

      if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong shape in model file: got [%d, %d], "
                "expected [%d, %d]\n",
                __func__, name.c_str(), (int)tensor->ne[0], (int)tensor->ne[1],
                ne[0], ne[1]);
        return false;
      }

      // for debugging
      if (0) {
        printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n",
               name.c_str(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)),
               ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
      }

      const size_t bpe = ggml_type_size(ggml_type(ttype));

      if ((nelements * bpe) / ggml_blck_size(tensor->type) !=
          ggml_nbytes(tensor)) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong size in model file: got %zu, "
                "expected %zu\n",
                __func__, name.c_str(), ggml_nbytes(tensor), nelements * bpe);
        return false;
      }

      if (ggml_backend_buffer_is_host(model.buffer_w)) {
        // for some backends such as CPU and Metal, the tensor data is in system
        // memory and we can read directly into it
        fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
      } else {
        // read into a temporary buffer first, then copy to device memory
        read_buf.resize(ggml_nbytes(tensor));
        fin.read(read_buf.data(), ggml_nbytes(tensor));
        ggml_backend_tensor_set(tensor, read_buf.data(), 0,
                                ggml_nbytes(tensor));
      }

      total_size += ggml_nbytes(tensor);
    }

    printf("%s: model size  = %8.2f MB\n", __func__,
           total_size / 1024.0 / 1024.0);
  }

  fin.close();

  return true;
}

// clang-format off

/*

 ██████╗ ██╗███████╗███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗
 ██╔══██╗██║██╔════╝██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║
 ██║  ██║██║█████╗  █████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║
 ██║  ██║██║██╔══╝  ██╔══╝  ██║   ██║╚════██║██║██║   ██║██║╚██╗██║
 ██████╔╝██║██║     ██║     ╚██████╔╝███████║██║╚██████╔╝██║ ╚████║
 ╚═════╝ ╚═╝╚═╝     ╚═╝      ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝

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

// clang-format on

// derived from  gpt2_model_load(const std::string & fname, gpt2_model & model,
// gpt_vocab & vocab, int n_ctx, int n_gpu_layers) {
bool diffusion_model_load(const std::string &fname, diffusion_model &model) {
  printf("%s: loading model from '%s'\n", __func__, fname.c_str());

  auto fin = std::ifstream(fname, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    return false;
  }

  // verify magic
  {
    uint32_t magic;
    fin.read((char *)&magic, sizeof(magic));
    if (magic != GGML_FILE_MAGIC) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__,
              fname.c_str());
      return false;
    }
  }

  size_t buffer_size = 0;

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1 * 2048); // conditioning latent

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024 * 3); // latent conditioning weight
  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // latent conditioning bias

  for (int i = 0; i < 4; i++) {
    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // latent conditioner attention block norm weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // latent conditioner attention block norm bias
    buffer_size += ggml_row_size(GGML_TYPE_F32, 3072 * 1024); // latent conditioner key value query weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 3072); // latent conditioner key value query bias
    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024); // latent conditioner projection out weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // latent conditioner projection out bias
    buffer_size += ggml_row_size(GGML_TYPE_F32, 16 * 32); // latent conditioner relative position embeddings
                            // relative attention bias weight
  }

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // code norm weight
  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // code norm bias

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024); // time embed linear 0 weight
  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // time embed linear 0 bias

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024); // time embed linear 1 weight
  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // time embed linear 1 bias

  // conditioning timestep integrator diffusion layers
  for (int i = 0; i < 3; i++) {

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // conditioning_timestep_integrator.0.resblk.in_layers.0.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // conditioning_timestep_integrator.0.resblk.in_layers.0.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024); // conditioning_timestep_integrator.0.resblk.in_layers.2.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // conditioning_timestep_integrator.0.resblk.in_layers.2.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 2048 * 1024); // conditioning_timestep_integrator.0.resblk.emb_layers.1.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 2048); // conditioning_timestep_integrator.0.resblk.emb_layers.1.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // conditioning_timestep_integrator.0.resblk.out_layers.0.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // conditioning_timestep_integrator.0.resblk.out_layers.0.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024 * 3); // conditioning_timestep_integrator.0.resblk.out_layers.3.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // conditioning_timestep_integrator.0.resblk.out_layers.3.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // conditioning_timestep_integrator.0.attn.norm.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // conditioning_timestep_integrator.0.attn.norm.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 3072 * 1024); // conditioning_timestep_integrator.0.attn.qkv.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 3072); // conditioning_timestep_integrator.0.attn.qkv.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024); // conditioning_timestep_integrator.0.attn.proj_out.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // conditioning_timestep_integrator.0.attn.proj_out.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 32 * 16); // conditioning_timestep_integrator.0.attn.relative_pos_embeddings.relative_attention_bias.weight
  }

  buffer_size += ggml_row_size(GGML_TYPE_F32, 3 * 100 * 1024);  // inp_block weight
  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // inp_block bias

  buffer_size += ggml_row_size(GGML_TYPE_F32, 2048 * 1024); // integrating_conv weight
  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // integrating conv bias

  // main diffusion layers
  for (int i = 0; i < 10; i++) {

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // resblk.in_layers.0.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // resblk.in_layers.0.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024); // resblk.in_layers.2.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // resblk.in_layers.2.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 2048 * 1024); // resblk.emb_layers.1.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 2048); // resblk.emb_layers.1.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // resblk.out_layers.0.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // resblk.out_layers.0.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024 * 3); // resblk.out_layers.3.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // resblk.out_layers.3.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // attn.norm.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // attn.norm.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 3072 * 1024); // attn.qkv.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 3072); // attn.qkv.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024); // attn.proj_out.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // attn.proj_out.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 32 * 16); // attn.relative_pos_embeddings.relative_attention_bias.weight
  }

  // main residual blocks
  for (int i = 0; i < 3; i++) {

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // in_layers.0.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // in_layers.0.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024); // in_layers.2.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // in_layers.2.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 2048 * 1024); // emb_layers.1.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 2048); // emb_layers.1.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // out_layers.0.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // out_layers.0.bias

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 1024 * 3); // out_layers.3.weight

    buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // out_layers.3.bias
  }

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // out group norm weight
  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // out group norm bias
  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 200 * 3); // out convolution weight
  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024 * 200); // out convolution bias

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1024); // unconditioned embedding

  printf("%s: ggml tensor size    = %d bytes\n", __func__,
         (int)sizeof(ggml_tensor));
  printf("%s: backend buffer size = %6.2f MB\n", __func__,
         buffer_size / (1024.0 * 1024.0));

  struct ggml_init_params params = {
      ggml_tensor_overhead() * (size_t)(5 + 7 * 4 + 4 + (3 * 18) + 4 +
                                        (10 * 18) + (3 * 10)), // mem size
      NULL,                                                    // mem buffer
      true,                                                    // no alloc
  };

  model.ctx = ggml_init(params);

  if (!model.ctx) {
    fprintf(stderr, "%s: ggml_init() failed\n", __func__);
    return false;
  }

  // initialize the backend
#ifdef GGML_USE_CUDA
  fprintf(stderr, "%s: using CUDA backend\n", __func__);
  model.backend = ggml_backend_cuda_init(0);
  if (!model.backend) {
    fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
  }
#endif

#ifdef GGML_USE_METAL
  fprintf(stderr, "%s: using Metal backend\n", __func__);
  // ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
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

  // model.buffer_w = ggml_backend_alloc_buffer(model.backend, buffer_size);

  auto &ctx = model.ctx;

  model.diffusion_conditioning_latent =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
  model.latent_conditioner_convolution_weight =
      ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1024, 1024);
  model.latent_conditioner_convolution_bias =
      ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
  model.code_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
  model.code_norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

  model.latent_conditioner_attention_blocks.resize(4);
  for (int i = 1; i < 5; i++) {

    auto &block = model.latent_conditioner_attention_blocks[i - 1];

    block.norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    block.norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    block.qkv_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 3072);
    block.qkv_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3072);
    block.projection_out_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
    block.projection_out_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    block.relative_position_embeddings_relative_attention_bias_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 32);

    model.tensors["latent_conditioner." + std::to_string(i) + ".norm.weight"] =
        block.norm_weight;
    model.tensors["latent_conditioner." + std::to_string(i) + ".norm.bias"] =
        block.norm_bias;
    model.tensors["latent_conditioner." + std::to_string(i) + ".qkv.weight"] =
        block.qkv_weight;
    model.tensors["latent_conditioner." + std::to_string(i) + ".qkv.bias"] =
        block.qkv_bias;
    model.tensors["latent_conditioner." + std::to_string(i) +
                  ".proj_out.weight"] = block.projection_out_weight;
    model
        .tensors["latent_conditioner." + std::to_string(i) + ".proj_out.bias"] =
        block.projection_out_bias;
    model.tensors["latent_conditioner." + std::to_string(i) +
                  ".relative_pos_embeddings.relative_attention_bias.weight"] =
        block.relative_position_embeddings_relative_attention_bias_weight;
  }

  model.time_emb_linear_0_weight =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
  model.time_emb_linear_0_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

  model.time_emb_linear_1_weight =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
  model.time_emb_layer_norm_1_bias =
      ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

  model.timestep_conditioning_integrator_diffusion_layers.resize(3);
  for (int i = 0; i < 3; i++) {
    auto &layer = model.timestep_conditioning_integrator_diffusion_layers[i];

    layer.resblock_in_layers_0_weight =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.resblock_in_layers_0_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.resblock_in_layers_2_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
    layer.resblock_in_layers_2_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.resblock_emb_layers_1_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 2048);
    layer.resblock_emb_layers_1_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    layer.resblock_out_layers_0_weight =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.resblock_out_layers_0_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.resblock_out_layers_3_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1024, 1024);
    layer.resblock_out_layers_3_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.attn_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.attn_norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.attn_qkv_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 3072);
    layer.attn_qkv_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3072);
    layer.attn_proj_out_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
    layer.attn_proj_out_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.attn_relative_pos_embeddings_relative_attention_bias_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 32);

    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".resblk.in_layers.0.weight"] =
        layer.resblock_in_layers_0_weight;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".resblk.in_layers.0.bias"] = layer.resblock_in_layers_0_bias;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".resblk.in_layers.2.weight"] =
        layer.resblock_in_layers_2_weight;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".resblk.in_layers.2.bias"] = layer.resblock_in_layers_2_bias;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".resblk.emb_layers.1.weight"] =
        layer.resblock_emb_layers_1_weight;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".resblk.emb_layers.1.bias"] =
        layer.resblock_emb_layers_1_bias;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".resblk.out_layers.0.weight"] =
        layer.resblock_out_layers_0_weight;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".resblk.out_layers.0.bias"] =
        layer.resblock_out_layers_0_bias;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".resblk.out_layers.3.weight"] =
        layer.resblock_out_layers_3_weight;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".resblk.out_layers.3.bias"] =
        layer.resblock_out_layers_3_bias;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".attn.norm.weight"] = layer.attn_norm_weight;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".attn.norm.bias"] = layer.attn_norm_bias;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".attn.qkv.weight"] = layer.attn_qkv_weight;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".attn.qkv.bias"] = layer.attn_qkv_bias;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".attn.proj_out.weight"] = layer.attn_proj_out_weight;
    model.tensors["conditioning_timestep_integrator." + std::to_string(i) +
                  ".attn.proj_out.bias"] = layer.attn_proj_out_bias;
    model.tensors
        ["conditioning_timestep_integrator." + std::to_string(i) +
         ".attn.relative_pos_embeddings.relative_attention_bias.weight"] =
        layer.attn_relative_pos_embeddings_relative_attention_bias_weight;
  }

  model.inp_block_weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 100, 1024);
  model.inp_block_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

  model.integrating_conv_weight =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1024);
  model.integrating_conv_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

  model.main_diffusion_layers.resize(10);
  for (int i = 0; i < 10; i++) {
    auto &layer = model.main_diffusion_layers[i];

    layer.resblock_in_layers_0_weight =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.resblock_in_layers_0_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.resblock_in_layers_2_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
    layer.resblock_in_layers_2_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.resblock_emb_layers_1_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 2048);
    layer.resblock_emb_layers_1_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    layer.resblock_out_layers_0_weight =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.resblock_out_layers_0_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.resblock_out_layers_3_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1024, 1024);
    layer.resblock_out_layers_3_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.attn_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.attn_norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.attn_qkv_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 3072);
    layer.attn_qkv_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3072);
    layer.attn_proj_out_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
    layer.attn_proj_out_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.attn_relative_pos_embeddings_relative_attention_bias_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 32);

    model
        .tensors["layers." + std::to_string(i) + ".resblk.in_layers.0.weight"] =
        layer.resblock_in_layers_0_weight;
    model.tensors["layers." + std::to_string(i) + ".resblk.in_layers.0.bias"] =
        layer.resblock_in_layers_0_bias;
    model
        .tensors["layers." + std::to_string(i) + ".resblk.in_layers.2.weight"] =
        layer.resblock_in_layers_2_weight;
    model.tensors["layers." + std::to_string(i) + ".resblk.in_layers.2.bias"] =
        layer.resblock_in_layers_2_bias;
    model.tensors["layers." + std::to_string(i) +
                  ".resblk.emb_layers.1.weight"] =
        layer.resblock_emb_layers_1_weight;
    model.tensors["layers." + std::to_string(i) + ".resblk.emb_layers.1.bias"] =
        layer.resblock_emb_layers_1_bias;
    model.tensors["layers." + std::to_string(i) +
                  ".resblk.out_layers.0.weight"] =
        layer.resblock_out_layers_0_weight;
    model.tensors["layers." + std::to_string(i) + ".resblk.out_layers.0.bias"] =
        layer.resblock_out_layers_0_bias;
    model.tensors["layers." + std::to_string(i) +
                  ".resblk.out_layers.3.weight"] =
        layer.resblock_out_layers_3_weight;
    model.tensors["layers." + std::to_string(i) + ".resblk.out_layers.3.bias"] =
        layer.resblock_out_layers_3_bias;
    model.tensors["layers." + std::to_string(i) + ".attn.norm.weight"] =
        layer.attn_norm_weight;
    model.tensors["layers." + std::to_string(i) + ".attn.norm.bias"] =
        layer.attn_norm_bias;
    model.tensors["layers." + std::to_string(i) + ".attn.qkv.weight"] =
        layer.attn_qkv_weight;
    model.tensors["layers." + std::to_string(i) + ".attn.qkv.bias"] =
        layer.attn_qkv_bias;
    model.tensors["layers." + std::to_string(i) + ".attn.proj_out.weight"] =
        layer.attn_proj_out_weight;
    model.tensors["layers." + std::to_string(i) + ".attn.proj_out.bias"] =
        layer.attn_proj_out_bias;
    model.tensors
        ["layers." + std::to_string(i) +
         ".attn.relative_pos_embeddings.relative_attention_bias.weight"] =
        layer.attn_relative_pos_embeddings_relative_attention_bias_weight;
  }

  model.main_residual_blocks.resize(3);
  for (int i = 10; i < 13; i++) {
    auto &layer = model.main_residual_blocks[i - 10];

    layer.in_layers_0_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.in_layers_0_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.in_layers_2_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
    layer.in_layers_2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.emb_layers_1_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 2048);
    layer.emb_layers_1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    layer.out_layers_0_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.out_layers_0_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    layer.out_layers_3_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1024, 1024);
    layer.out_layers_3_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

    model.tensors["layers." + std::to_string(i) + ".in_layers.0.weight"] =
        layer.in_layers_0_weight;
    model.tensors["layers." + std::to_string(i) + ".in_layers.0.bias"] =
        layer.in_layers_0_bias;
    model.tensors["layers." + std::to_string(i) + ".in_layers.2.weight"] =
        layer.in_layers_2_weight;
    model.tensors["layers." + std::to_string(i) + ".in_layers.2.bias"] =
        layer.in_layers_2_bias;
    model.tensors["layers." + std::to_string(i) + ".emb_layers.1.weight"] =
        layer.emb_layers_1_weight;
    model.tensors["layers." + std::to_string(i) + ".emb_layers.1.bias"] =
        layer.emb_layers_1_bias;
    model.tensors["layers." + std::to_string(i) + ".out_layers.0.weight"] =
        layer.out_layers_0_weight;
    model.tensors["layers." + std::to_string(i) + ".out_layers.0.bias"] =
        layer.out_layers_0_bias;
    model.tensors["layers." + std::to_string(i) + ".out_layers.3.weight"] =
        layer.out_layers_3_weight;
    model.tensors["layers." + std::to_string(i) + ".out_layers.3.bias"] =
        layer.out_layers_3_bias;
  }

  model.out_group_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
  model.out_group_norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

  model.out_convolution_weight =
      ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1024, 200);
  model.out_convolution_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 200);

  model.unconditioned_embedding = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

  model.tensors["unconditioned_embedding"] = model.unconditioned_embedding;

  model.tensors["out.0.weight"] = model.out_group_norm_weight;
  model.tensors["out.0.bias"] = model.out_group_norm_bias;

  model.tensors["out.2.weight"] = model.out_convolution_weight;
  model.tensors["out.2.bias"] = model.out_convolution_bias;

  model.tensors["diffusion_conditioning_latent"] =
      model.diffusion_conditioning_latent;
  model.tensors["latent_conditioner.0.bias"] =
      model.latent_conditioner_convolution_bias;
  model.tensors["latent_conditioner.0.weight"] =
      model.latent_conditioner_convolution_weight;

  model.tensors["code_norm.weight"] = model.code_norm_weight;
  model.tensors["code_norm.bias"] = model.code_norm_bias;

  model.tensors["time_embed.0.weight"] = model.time_emb_linear_0_weight;
  model.tensors["time_embed.0.bias"] = model.time_emb_linear_0_bias;

  model.tensors["time_embed.2.weight"] = model.time_emb_linear_1_weight;
  model.tensors["time_embed.2.bias"] = model.time_emb_layer_norm_1_bias;

  model.tensors["inp_block.weight"] = model.inp_block_weight;
  model.tensors["inp_block.bias"] = model.inp_block_bias;

  model.tensors["integrating_conv.weight"] = model.integrating_conv_weight;
  model.tensors["integrating_conv.bias"] = model.integrating_conv_bias;

  {
    // ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer_w);
    model.buffer_w = ggml_backend_alloc_ctx_tensors(ctx, model.backend);

    size_t total_size = 0;

    bool has_lm_head = false;

    std::vector<char> read_buf;

    while (true) {
      int32_t n_dims;
      int32_t length;
      int32_t ttype;

      fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
      fin.read(reinterpret_cast<char *>(&length), sizeof(length));
      fin.read(reinterpret_cast<char *>(&ttype), sizeof(ttype));

      if (fin.eof()) {
        break;
      }

      int32_t nelements = 1;
      int32_t ne[4] = {1, 1, 1, 1};
      for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
      }

      std::string name(length, 0);
      fin.read(&name[0], length);

      if (model.tensors.find(name) == model.tensors.end()) {
        fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__,
                name.c_str());
        return false;
      }

      auto tensor = model.tensors[name];
      ggml_set_name(tensor, name.c_str());
      if (ggml_nelements(tensor) != nelements) {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n",
                __func__, name.c_str());
        return false;
      }

      if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong shape in model file: got [%d, %d], "
                "expected [%d, %d]\n",
                __func__, name.c_str(), (int)tensor->ne[0], (int)tensor->ne[1],
                ne[0], ne[1]);
        return false;
      }

      // for debugging
      if (0) {
        printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n",
               name.c_str(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)),
               ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
      }

      const size_t bpe = ggml_type_size(ggml_type(ttype));

      if ((nelements * bpe) / ggml_blck_size(tensor->type) !=
          ggml_nbytes(tensor)) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong size in model file: got %zu, "
                "expected %zu\n",
                __func__, name.c_str(), ggml_nbytes(tensor), nelements * bpe);
        return false;
      }

      if (ggml_backend_buffer_is_host(model.buffer_w)) {
        // for some backends such as CPU and Metal, the tensor data is in system
        // memory and we can read directly into it
        fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
      } else {
        // read into a temporary buffer first, then copy to device memory
        read_buf.resize(ggml_nbytes(tensor));
        fin.read(read_buf.data(), ggml_nbytes(tensor));
        ggml_backend_tensor_set(tensor, read_buf.data(), 0,
                                ggml_nbytes(tensor));
      }

      total_size += ggml_nbytes(tensor);
    }

    printf("%s: model size  = %8.2f MB\n", __func__,
           total_size / 1024.0 / 1024.0);
  }

  fin.close();

  return true;
}

// clang-format off

/*

 ██╗   ██╗ ██████╗  ██████╗ ██████╗ ██████╗ ███████╗██████╗
 ██║   ██║██╔═══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗
 ██║   ██║██║   ██║██║     ██║   ██║██║  ██║█████╗  ██████╔╝
 ╚██╗ ██╔╝██║   ██║██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗
  ╚████╔╝ ╚██████╔╝╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║
   ╚═══╝   ╚═════╝  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝

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

// clang-format on

bool vocoder_model_load(const std::string &fname, vocoder_model &model) {
  printf("%s: loading model from '%s'\n", __func__, fname.c_str());

  auto fin = std::ifstream(fname, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    return false;
  }

  // verify magic
  {
    uint32_t magic;
    fin.read((char *)&magic, sizeof(magic));
    if (magic != GGML_FILE_MAGIC) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__,
              fname.c_str());
      return false;
    }
  }

  size_t buffer_size = 0;

  buffer_size += ggml_row_size(GGML_TYPE_F32, 32); // pre convolution bias
  buffer_size += ggml_row_size(GGML_TYPE_F32, 32 * 64 * 7); // pre convolution weight

  for (int i = 0; i < 3; i++) {
    buffer_size += ggml_row_size(GGML_TYPE_F32, 64); // kernel_predictor.input_conv.0.bias
    buffer_size += ggml_row_size(GGML_TYPE_F32, 64 * 100 * 5); // kernel_predictor.input_conv.0.weight

    for (int c = 0; c < 3; c++) {

      buffer_size += ggml_row_size(GGML_TYPE_F32, 64); // kernel_predictor.residual_convs.0.1.bias
      buffer_size += ggml_row_size(GGML_TYPE_F32, 64 * 64 * 3); // kernel_predictor.residual_convs.0.1.weight
      buffer_size += ggml_row_size(GGML_TYPE_F32, 64); // kernel_predictor.residual_convs.0.3.bias
      buffer_size += ggml_row_size(GGML_TYPE_F32, 64 * 64 * 3); // kernel_predictor.residual_convs.0.3.weight
    }

    buffer_size += ggml_row_size(GGML_TYPE_F32, 24576); // kernel_predictor.kernel_conv.bias
    buffer_size += ggml_row_size(GGML_TYPE_F32, 24576 * 64 * 3); // kernel_predictor.kernel_conv.weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 256); // kernel_predictor.bias_conv.bias
    buffer_size += ggml_row_size(GGML_TYPE_F32, 256 * 64 * 3); // kernel_predictor.bias_conv.weight

    if (i < 2) {
      buffer_size += ggml_row_size(GGML_TYPE_F32, 32 * 32 * 16); // convt_pre.1.weight
    } else {
      buffer_size += ggml_row_size(GGML_TYPE_F32, 32 * 32 * 8); // convt_pre.1.weight
    }

    buffer_size += ggml_row_size(GGML_TYPE_F32, 32); // convt_pre.1.bias
    buffer_size += ggml_row_size(GGML_TYPE_F32, 32 * 32 * 3); // conv_blocks.0.1.weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 32); // conv_blocks.0.1.bias
    buffer_size += ggml_row_size(GGML_TYPE_F32, 32 * 32 * 3); // conv_blocks.1.1.weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 32); // conv_blocks.1.1.bias
    buffer_size += ggml_row_size(GGML_TYPE_F32, 32 * 32 * 3); // conv_blocks.2.1.weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 32); // conv_blocks.2.1.bias
    buffer_size += ggml_row_size(GGML_TYPE_F32, 32 * 32 * 3); // conv_blocks.3.1.weight
    buffer_size += ggml_row_size(GGML_TYPE_F32, 32); // conv_blocks.3.1.bias
  }

  buffer_size += ggml_row_size(GGML_TYPE_F32, 1); // post convolution bias
  buffer_size += ggml_row_size(GGML_TYPE_F32, 32 * 7); // post convolution weight

  printf("%s: ggml tensor size    = %d bytes\n", __func__,
         (int)sizeof(ggml_tensor));
  printf("%s: backend buffer size = %6.2f MB\n", __func__,
         buffer_size / (1024.0 * 1024.0));

  struct ggml_init_params params = {
      ggml_tensor_overhead() * (size_t)(177), // mem size
      NULL,                                   // mem buffer
      true,                                   // no alloc
  };

  model.ctx = ggml_init(params);

  if (!model.ctx) {
    fprintf(stderr, "%s: ggml_init() failed\n", __func__);
    return false;
  }

  // initialize the backend
#ifdef GGML_USE_CUDA
  fprintf(stderr, "%s: using CUDA backend\n", __func__);
  model.backend = ggml_backend_cuda_init(0);
  if (!model.backend) {
    fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
  }
#endif

#ifdef GGML_USE_METAL
  fprintf(stderr, "%s: using Metal backend\n", __func__);
  // ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
  model.backend = ggml_backend_metal_init();
  if (!model.backend) {
    fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
  }

#endif

  if (!model.backend) {
    // fallback to CPU backend
    fprintf(stderr, "%s: using CPU backend\n", __func__);
    model.backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(model.backend, 1); // FIXME: ggml crashes with more than 1 thread at the end of procesing
  }

  if (!model.backend) {
    fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
    return false;
  }

  // model.buffer_w = ggml_backend_alloc_buffer(model.backend, buffer_size);

  auto &ctx = model.ctx;

  model.convolution_pre_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
  model.convolution_pre_weight =
      ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 64, 32);

  model.residual_stack.resize(3);
  for (int i = 0; i < 3; i++) {
    model.residual_stack[i].kernel_predictor_input_convolution_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    model.residual_stack[i].kernel_predictor_input_convolution_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 100, 64);

    model.residual_stack[i].kernel_predictor_residual_conv_blocks.resize(3);
    for (int c = 0; c < 3; c++) {
      model.residual_stack[i]
          .kernel_predictor_residual_conv_blocks[c]
          .residual_convs_1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
      model.residual_stack[i]
          .kernel_predictor_residual_conv_blocks[c]
          .residual_convs_1_weight =
          ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 64, 64);
      model.residual_stack[i]
          .kernel_predictor_residual_conv_blocks[c]
          .residual_convs_3_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
      model.residual_stack[i]
          .kernel_predictor_residual_conv_blocks[c]
          .residual_convs_3_weight =
          ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 64, 64);

      model.tensors["res_stack." + std::to_string(i) +
                    ".kernel_predictor.residual_convs." + std::to_string(c) +
                    ".1.bias"] = model.residual_stack[i]
                                     .kernel_predictor_residual_conv_blocks[c]
                                     .residual_convs_1_bias;
      model.tensors["res_stack." + std::to_string(i) +
                    ".kernel_predictor.residual_convs." + std::to_string(c) +
                    ".1.weight"] = model.residual_stack[i]
                                       .kernel_predictor_residual_conv_blocks[c]
                                       .residual_convs_1_weight;
      model.tensors["res_stack." + std::to_string(i) +
                    ".kernel_predictor.residual_convs." + std::to_string(c) +
                    ".3.bias"] = model.residual_stack[i]
                                     .kernel_predictor_residual_conv_blocks[c]
                                     .residual_convs_3_bias;
      model.tensors["res_stack." + std::to_string(i) +
                    ".kernel_predictor.residual_convs." + std::to_string(c) +
                    ".3.weight"] = model.residual_stack[i]
                                       .kernel_predictor_residual_conv_blocks[c]
                                       .residual_convs_3_weight;
    }

    model.residual_stack[i].kernel_predictor_kernel_convolution_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 24576);
    model.residual_stack[i].kernel_predictor_kernel_convolution_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 64, 24576);
    model.residual_stack[i].kernel_predictor_bias_convolution_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    model.residual_stack[i].kernel_predictor_bias_convolution_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 64, 256);

    if (i < 2) {
      model.residual_stack[i].convolution_t_pre_weight =
          ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 32, 32);
    } else {
      model.residual_stack[i].convolution_t_pre_weight =
          ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 32, 32);
    }
    model.residual_stack[i].convolution_t_pre_bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);

    model.tensors["res_stack." + std::to_string(i) +
                  ".kernel_predictor.input_conv.0.bias"] =
        model.residual_stack[i].kernel_predictor_input_convolution_bias;
    model.tensors["res_stack." + std::to_string(i) +
                  ".kernel_predictor.input_conv.0.weight"] =
        model.residual_stack[i].kernel_predictor_input_convolution_weight;
    model.tensors["res_stack." + std::to_string(i) +
                  ".kernel_predictor.kernel_conv.bias"] =
        model.residual_stack[i].kernel_predictor_kernel_convolution_bias;
    model.tensors["res_stack." + std::to_string(i) +
                  ".kernel_predictor.kernel_conv.weight"] =
        model.residual_stack[i].kernel_predictor_kernel_convolution_weight;
    model.tensors["res_stack." + std::to_string(i) +
                  ".kernel_predictor.bias_conv.bias"] =
        model.residual_stack[i].kernel_predictor_bias_convolution_bias;
    model.tensors["res_stack." + std::to_string(i) +
                  ".kernel_predictor.bias_conv.weight"] =
        model.residual_stack[i].kernel_predictor_bias_convolution_weight;
    model.tensors["res_stack." + std::to_string(i) + ".convt_pre.1.weight"] =
        model.residual_stack[i].convolution_t_pre_weight;
    model.tensors["res_stack." + std::to_string(i) + ".convt_pre.1.bias"] =
        model.residual_stack[i].convolution_t_pre_bias;

    model.residual_stack[i].conv_blocks.resize(4);

    for (int c = 0; c < 4; c++) {
      model.residual_stack[i].conv_blocks[c].conv_block_1_weight =
          ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 32, 32);
      model.residual_stack[i].conv_blocks[c].conv_block_1_bias =
          ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);

      model.tensors["res_stack." + std::to_string(i) + ".conv_blocks." +
                    std::to_string(c) + ".1.weight"] =
          model.residual_stack[i].conv_blocks[c].conv_block_1_weight;
      model.tensors["res_stack." + std::to_string(i) + ".conv_blocks." +
                    std::to_string(c) + ".1.bias"] =
          model.residual_stack[i].conv_blocks[c].conv_block_1_bias;
    }
  }

  model.convolution_post_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  model.convolution_post_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 7, 32);

  model.tensors["conv_pre.bias"] = model.convolution_pre_bias;
  model.tensors["conv_pre.weight"] = model.convolution_pre_weight;
  model.tensors["conv_post.1.bias"] = model.convolution_post_bias;
  model.tensors["conv_post.1.weight"] = model.convolution_post_weight;

  {
    // ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer_w);
    model.buffer_w = ggml_backend_alloc_ctx_tensors(ctx, model.backend);

    size_t total_size = 0;

    bool has_lm_head = false;

    std::vector<char> read_buf;

    while (true) {
      int32_t n_dims;
      int32_t length;
      int32_t ttype;

      fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
      fin.read(reinterpret_cast<char *>(&length), sizeof(length));
      fin.read(reinterpret_cast<char *>(&ttype), sizeof(ttype));

      if (fin.eof()) {
        break;
      }

      int32_t nelements = 1;
      int32_t ne[4] = {1, 1, 1, 1};
      for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
      }

      std::string name(length, 0);
      fin.read(&name[0], length);

      if (model.tensors.find(name) == model.tensors.end()) {
        fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__,
                name.c_str());
        return false;
      }

      auto tensor = model.tensors[name];
      ggml_set_name(tensor, name.c_str());
      if (ggml_nelements(tensor) != nelements) {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n",
                __func__, name.c_str());
        return false;
      }

      if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong shape in model file: got [%d, %d], "
                "expected [%d, %d]\n",
                __func__, name.c_str(), (int)tensor->ne[0], (int)tensor->ne[1],
                ne[0], ne[1]);
        return false;
      }

      // for debugging
      if (0) {
        printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n",
               name.c_str(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)),
               ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
      }

      const size_t bpe = ggml_type_size(ggml_type(ttype));

      if ((nelements * bpe) / ggml_blck_size(tensor->type) !=
          ggml_nbytes(tensor)) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong size in model file: got %zu, "
                "expected %zu\n",
                __func__, name.c_str(), ggml_nbytes(tensor), nelements * bpe);
        return false;
      }

      if (ggml_backend_buffer_is_host(model.buffer_w)) {
        // for some backends such as CPU and Metal, the tensor data is in system
        // memory and we can read directly into it
        fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
      } else {
        // read into a temporary buffer first, then copy to device memory
        read_buf.resize(ggml_nbytes(tensor));
        fin.read(read_buf.data(), ggml_nbytes(tensor));
        ggml_backend_tensor_set(tensor, read_buf.data(), 0,
                                ggml_nbytes(tensor));
      }

      total_size += ggml_nbytes(tensor);
    }

    printf("%s: model size  = %8.2f MB\n", __func__,
           total_size / 1024.0 / 1024.0);
  }

  fin.close();

  return true;
}

// clang-format off

/*
 
  ██████╗ ██████╗ ████████╗   ██████╗                
 ██╔════╝ ██╔══██╗╚══██╔══╝   ╚════██╗               
 ██║  ███╗██████╔╝   ██║█████╗ █████╔╝               
 ██║   ██║██╔═══╝    ██║╚════╝██╔═══╝                
 ╚██████╔╝██║        ██║      ███████╗               
  ╚═════╝ ╚═╝        ╚═╝      ╚══════╝               
                                                     
 ██╗      █████╗ ████████╗███████╗███╗   ██╗████████╗
 ██║     ██╔══██╗╚══██╔══╝██╔════╝████╗  ██║╚══██╔══╝
 ██║     ███████║   ██║   █████╗  ██╔██╗ ██║   ██║   
 ██║     ██╔══██║   ██║   ██╔══╝  ██║╚██╗██║   ██║   
 ███████╗██║  ██║   ██║   ███████╗██║ ╚████║   ██║   
 ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝   ╚═╝   
                                                     
  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗           
 ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║           
 ██║  ███╗██████╔╝███████║██████╔╝███████║           
 ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║           
 ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║           
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝           
                                                     
 
*/

// clang-format on

struct ggml_cgraph *autoregressive_latent_graph(
    const autoregressive_model &model,
    const std::vector<int> mel_transformer_inputs_vector,
    const std::vector<gpt_vocab::id> &tokens, const int batch_size) {
  const int token_count = tokens.size();
  // std::cout << "token count: " << token_count << std::endl;
  // exit(0);
  const int mel_token_count = mel_transformer_inputs_vector.size();

  static size_t buf_size = ggml_tensor_overhead() * GPT2_MAX_NODES +
                           ggml_graph_overhead_custom(GPT2_MAX_NODES, false);
  static std::vector<uint8_t> buf(buf_size);

  struct ggml_init_params params = {
      /*.mem_size   =*/buf_size,
      /*.mem_buffer =*/buf.data(),
      /*.no_alloc   =*/true, // the tensors will be allocated later by
                             // ggml_gallocr_alloc_graph()
  };

  struct ggml_context *ctx0 = ggml_init(params);

  struct ggml_cgraph *gf = ggml_new_graph_custom(ctx0, GPT2_MAX_NODES, false);

  struct ggml_tensor *input =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_count * batch_size);

  ggml_set_name(input, "input_tokens");

  struct ggml_tensor *conditioning_latent =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1024);

  ggml_set_name(conditioning_latent, "auto_conditioning");

  ggml_build_forward_expand(gf, input);

  // ggml_set_name(input, "text input codes");

  struct ggml_tensor *input_position =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_count * batch_size);

  ggml_set_name(input_position, "input_position");

  struct ggml_tensor *mel_codes =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, mel_token_count);

  ggml_set_name(mel_codes, "input_mel_tokens");

  // ggml_tensor * temp_cur = ggml_cpy(ctx0, mel_codes, ggml_new_tensor(ctx0,
  // GGML_TYPE_I32,4,mel_codes->ne) );

  struct ggml_tensor *mel_embedding =
      ggml_get_rows(ctx0, model.mel_embedding_weights, mel_codes);

  // struct ggml_tensor * mel_embedding = ggml_get_rows(ctx0,
  // model.mel_embedding_weights,input);

  ggml_set_name(model.mel_embedding_weights, "mel emb weights");
  ggml_set_name(mel_embedding, "mel embedding");

  struct ggml_tensor *mel_position =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, mel_token_count);
  ggml_set_name(mel_position, "input_mel_position");

  struct ggml_tensor *mel_position_embedding =
      ggml_get_rows(ctx0, model.mel_position_embedding_weights, mel_position);

  ggml_set_name(mel_position_embedding, "mel position embedding");

  mel_embedding = ggml_add(ctx0, mel_embedding, mel_position_embedding);

  struct ggml_tensor *text_embedding =
      ggml_get_rows(ctx0, model.text_embedding_weights, input);
  struct ggml_tensor *text_position_embedding = ggml_get_rows(
      ctx0, model.text_position_embedding_weights, input_position);

  struct ggml_tensor *embedding =
      ggml_add(ctx0, text_embedding, text_position_embedding);
  ggml_set_name(embedding, "final text embedding");

  ggml_build_forward_expand(gf, embedding);
  ggml_build_forward_expand(gf, mel_embedding);

  embedding =
      ggml_reshape_4d(ctx0, embedding, 1024, token_count, batch_size, 1);
  mel_embedding = ggml_reshape_4d(ctx0, mel_embedding, 1024,
                                  mel_token_count / batch_size, batch_size, 1);

  embedding = ggml_cont(ctx0, ggml_permute(ctx0, embedding, 0, 2, 1, 3));
  mel_embedding =
      ggml_cont(ctx0, ggml_permute(ctx0, mel_embedding, 0, 2, 1,
                                   3)); // removing the ggml_cont changes
                                        // behavior, looks like a ggml bug lol

  struct ggml_tensor *output = ggml_concat(ctx0, embedding, mel_embedding, 2);

  // output = ggml_cont(ctx0,ggml_permute(ctx0, output, 0,2,1,3));
  output = ggml_cont(ctx0, output);

  struct ggml_tensor *reshaped_latent =
      ggml_reshape_1d(ctx0, conditioning_latent, 1024);

  struct ggml_tensor *repeated_latent =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, batch_size * 1024);

  // repeated_latent = ggml_reshape_4d(ctx0, repeated_latent,1024, 1, 4, 1);

  repeated_latent = ggml_repeat(ctx0, reshaped_latent, repeated_latent);

  repeated_latent = ggml_cont(
      ctx0, ggml_reshape_4d(ctx0, repeated_latent, 1024, batch_size, 1, 1));

  output = ggml_concat(ctx0, repeated_latent, output, 2);

  ggml_set_name(output, "output");

  int test_dimension = output->ne[2];
  int n_past = 0;

  struct ggml_tensor *cur =
      ggml_cont(ctx0, ggml_permute(ctx0, output, 0, 2, 1, 3));

  struct ggml_tensor *Qcur;
  struct ggml_tensor *Kcur;
  struct ggml_tensor *Vcur;

  struct ggml_tensor *Q;
  struct ggml_tensor *K;

  struct ggml_tensor *KQ;
  struct ggml_tensor *KQ_scaled;
  struct ggml_tensor *KQ_masked;
  struct ggml_tensor *KQ_soft_max;
  struct ggml_tensor *V_transposed;
  struct ggml_tensor *KQV;
  struct ggml_tensor *KQV_merged;

  struct ggml_tensor *residual;
  struct ggml_tensor *feed_forward_residual;

  struct ggml_tensor *test;

  for (int i = 0; i < 30; i++) {

    residual = ggml_cpy(ctx0, cur,
                        ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1024,
                                           test_dimension, batch_size, 1));

    // ggml_build_forward_expand(gf, residual);
    // layer norm

    cur = ggml_norm(ctx0, cur, 1e-05);

    ggml_tensor *temp_cur =
        ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

    ggml_set_name(temp_cur, "postnorm");
    ggml_build_forward_expand(gf, temp_cur);

    ggml_format_name(cur, "l%d.norm", i);

    ggml_tensor *temp_ln_1_weights =
        ggml_repeat(ctx0, model.layers[i].layer_norm_1_weights,
                    ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

    ggml_set_name(temp_ln_1_weights, "weights");
    // ggml_build_forward_expand(gf, temp_ln_1_weights);

    cur = ggml_mul(
        ctx0, cur,
        temp_ln_1_weights); // if you flip the order of this it doesn't work on
                            // the second token generation process.TODO why does
                            // flipping the order of this break it?

    cur = ggml_add(ctx0, cur, model.layers[i].layer_norm_1_bias);

    // ggml_tensor * temp_weights = ggml_cpy(ctx0,
    // model.layers[i].layer_norm_1_weights, ggml_new_tensor(ctx0,
    // GGML_TYPE_F32,4,model.layers[i].layer_norm_1_weights->ne) );
    ggml_tensor *temp_bias =
        ggml_cpy(ctx0, model.layers[i].layer_norm_1_bias,
                 ggml_new_tensor(ctx0, GGML_TYPE_F32, 4,
                                 model.layers[i].layer_norm_1_bias->ne));

    //  if(!fake_inputs)
    // {
    // ggml_build_forward_expand(gf, temp_bias);
    // ggml_set_name(temp_bias, "bias");
    // return gf;
    // }

    // this is implemented as conv1d in pytorch, but it's actually just a affine
    // transformation with a weight and bias
    cur = ggml_mul_mat(
        ctx0,
        ggml_reshape_2d(
            ctx0,
            ggml_cont(ctx0,
                      ggml_transpose(
                          ctx0, model.layers[i].c_attention_attention_weights)),
            1024, 3072),
        cur);

    cur = ggml_reshape_4d(ctx0, cur, 3072, test_dimension, batch_size, 1);

    cur = ggml_add(
        ctx0, cur,
        model.layers[i]
            .c_attention_attention_bias); // worth studying the diffs here, why
                                          // did I have to remove this repeat
                                          // for settings where the second
                                          // dimension is 1?

    // derived from ggml reference gpt-2 implementation
    Qcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, 1024, test_dimension,
                                        batch_size, cur->nb[1], cur->nb[2], 0));

    // Kcur = ggml_cont(ctx0,ggml_permute(ctx0,ggml_view_4d(ctx0, cur, 1024,
    // test_dimension, 4, 1,cur->nb[1], cur->nb[2],cur->nb[3], 1024 *
    // sizeof(float)),0,1,3,2));

    Kcur = ggml_cont(
        ctx0,
        ggml_permute(ctx0,
                     ggml_view_3d(ctx0, cur, 1024, test_dimension, batch_size,
                                  cur->nb[1], cur->nb[2], 1024 * sizeof(float)),
                     0, 2, 1, 3));
    Vcur = ggml_cont(
        ctx0,
        ggml_permute(ctx0,
                     ggml_view_3d(ctx0, cur, 1024, test_dimension, batch_size,
                                  cur->nb[1], cur->nb[2], 2048 * sizeof(float)),
                     0, 2, 1, 3));

    //  struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_key, 1024 *
    //  test_dimension * 4 , (ggml_element_size(model.memory_key)* ((i * 404 *
    //  1024 * 4) + ((n_past) * 1024 *4)))); ggml_build_forward_expand(gf,
    //  ggml_cpy(ctx0, Kcur, k));

    //  struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_value, 1024 *
    //  test_dimension * 4 , (ggml_element_size(model.memory_value)* ((i * 404 *
    //  1024 * 4) + ((n_past) * 1024 *4)))); ggml_build_forward_expand(gf,
    //  ggml_cpy(ctx0, Vcur, v));

    // num heads 16
    // head dim 64

    Q = ggml_cont(ctx0,
                  ggml_permute(ctx0,
                               ggml_reshape_4d(ctx0, Qcur, 64, 16,
                                               test_dimension, batch_size),
                               0, 2, 1, 3));

    // this is likely not general and but should work for the first layer, may
    // need generalizing once we reach
    // the end of the first layer.
    // K =ggml_cont(ctx0,ggml_permute(ctx0,
    //             ggml_reshape_4d(ctx0, ggml_view_1d(ctx0, model.memory_key,
    //             1024 * (test_dimension + n_past) * 4 ,
    //             (ggml_element_size(model.memory_key)* (i * 404 * 1024 * 4) ))
    //             , 64,16, 4 , test_dimension + n_past), 0, 2,3, 1));
    K = ggml_cont(ctx0,
                  ggml_permute(ctx0,
                               ggml_reshape_4d(ctx0, Kcur, 64, 16, batch_size,
                                               test_dimension),
                               0, 2, 3, 1));

    V_transposed = ggml_cont(
        ctx0, ggml_permute(ctx0,
                           ggml_reshape_4d(ctx0, Vcur, 64, 16, batch_size,
                                           test_dimension),
                           1, 2, 3, 0));

    //   V_transposed = ggml_cont(ctx0,ggml_permute(ctx0,
    //               ggml_reshape_4d(ctx0, ggml_view_1d(ctx0,
    //               model.memory_value, 1024 * (test_dimension + n_past) * 4 ,
    //               (ggml_element_size(model.memory_value)* (i * 404 * 1024 *
    //               4) )) , 64,16,4,test_dimension+n_past), 1,2,3,0));

    // std::cout << K->ne[0]  << K->ne[1] << K->ne[2]  << K->ne[3] << std::endl;
    // std::cout << Q->ne[0]  << Q->ne[1] << Q->ne[2]  << Q->ne[3] << std::endl;
    // std::cout << V_transposed->ne[0]  << V_transposed->ne[1] <<
    // V_transposed->ne[2]  << V_transposed->ne[3] << std::endl;
    KQ = ggml_mul_mat(ctx0, K, Q);

    // std::cout << "KQ shape" << std::endl;
    // std::cout << KQ->ne[0]  << KQ->ne[1] << KQ->ne[2]  << KQ->ne[3] <<
    // std::endl;

    // KQ = ggml_reshape_1d(ctx0, KQ, KQ->ne[0]* KQ->ne[1]*KQ->ne[2]*KQ->ne[3]);

    KQ_scaled = ggml_scale_inplace(ctx0, KQ, 1.0f / sqrt(float(64)));

    KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

    KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

    KQV = ggml_mul_mat(ctx0, KQ_soft_max, V_transposed);

    // KQV = ggml_mul_mat(ctx0,
    // ggml_reshape_4d(ctx0,ggml_cont(ctx0,ggml_reshape_3d(ctx0, KQ_soft_max,
    // test_dimension + n_past,test_dimension + n_past,64)),test_dimension +
    // n_past,test_dimension+n_past,16,4),
    // ggml_reshape_3d(ctx0,V_transposed,n_past + test_dimension, 16,);

    ggml_set_name(KQ_soft_max, "after KQ");
    ggml_set_name(V_transposed, "after V");

    // getting the initial KQV value
    KQV = ggml_reshape_3d(ctx0, KQV, test_dimension, 64, 16 * batch_size);
    KQV = ggml_permute(ctx0, KQV, 1, 0, 2, 3);
    KQV = ggml_cont_3d(ctx0, KQV, 64, test_dimension, 16 * batch_size);
    KQV = ggml_reshape_4d(ctx0, KQV, 64, test_dimension, 16, batch_size);

    //"merge heads" operation
    KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
    KQV_merged =
        ggml_cont_3d(ctx0, KQV_merged, 1024, test_dimension, batch_size);

    cur = ggml_mul_mat(
        ctx0,
        ggml_reshape_2d(
            ctx0,
            ggml_cont(
                ctx0,
                ggml_transpose(ctx0,
                               model.layers[i].c_attention_projection_weights)),
            1024, 1024),
        KQV_merged);

    cur = ggml_add(ctx0, cur, model.layers[i].c_attention_projection_bias);

    // layer input passthrough
    // cur = ggml_add(ctx0, cur,residual);

    cur = ggml_add(
        ctx0, ggml_reshape_1d(ctx0, cur, 1024 * batch_size * test_dimension),
        ggml_reshape_1d(
            ctx0, residual,
            1024 * batch_size *
                test_dimension)); // it's really strange that this is necessary,
                                  // why does it have different behavior than
                                  // commented out above
    cur = ggml_reshape_4d(ctx0, cur, 1024, test_dimension, batch_size, 1);

    feed_forward_residual =
        ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

    // layer norm 2
    cur = ggml_norm(ctx0, cur, 1e-05);

    ggml_format_name(cur, "l%d.norm_2", i);

    ggml_tensor *temp_ln_2_weights =
        ggml_repeat(ctx0, model.layers[i].layer_norm_2_weights,
                    ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

    ggml_set_name(temp_ln_2_weights, "test");

    ggml_build_forward_expand(gf, temp_ln_2_weights);

    cur = ggml_mul(ctx0, cur, temp_ln_2_weights);
    cur = ggml_add(ctx0, cur, model.layers[i].layer_norm_2_bias);

    //  fully connected multi layer perceptron
    cur = ggml_mul_mat(
        ctx0,
        ggml_reshape_2d(
            ctx0,
            ggml_cont(
                ctx0,
                ggml_transpose(
                    ctx0,
                    model.layers[i]
                        .c_multi_layer_perceptron_fully_connected_weights)),
            1024, 4096),
        cur);

    cur =
        ggml_add(ctx0, cur,
                 model.layers[i].c_multi_layer_perceptron_fully_connected_bias);

    // gelu
    cur = ggml_gelu(ctx0, cur);

    // mlp fully connected
    cur = ggml_mul_mat(
        ctx0,
        ggml_reshape_2d(
            ctx0,
            ggml_cont(
                ctx0,
                ggml_transpose(
                    ctx0, model.layers[i]
                              .c_multi_layer_perceptron_projection_weights)),
            4096, 1024),
        cur);

    cur = ggml_add(ctx0, cur,
                   model.layers[i].c_multi_layer_perceptron_projection_bias);

    // final residual
    // another case where I had to flatten before adding to get correct results.
    // Either ggml had a pre-existing bug for batch addition, or one of my
    // modifications introduced it. This will need to be addressed.

    cur = ggml_add(
        ctx0,
        ggml_reshape_1d(ctx0, cur, 1024 * test_dimension * batch_size * 1),
        ggml_reshape_1d(ctx0, feed_forward_residual,
                        1024 * test_dimension * batch_size * 1));

    cur = ggml_reshape_4d(ctx0, cur, 1024, test_dimension, batch_size, 1);
    // cur = ggml_add(ctx0, cur, feed_forward_residual);
  }

  cur = ggml_norm(ctx0, cur, 1e-05);

  ggml_tensor *temp_final_layer_norm_weights =
      ggml_repeat(ctx0, model.final_layer_norm_weights,
                  ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

  ggml_build_forward_expand(gf, temp_final_layer_norm_weights);

  cur = ggml_mul(ctx0, cur, temp_final_layer_norm_weights);
  cur = ggml_add(ctx0, cur, model.final_layer_norm_bias);

  cur = ggml_norm(ctx0, cur, 1e-05);

  cur = ggml_cont(ctx0, ggml_view_4d(ctx0, cur, 1024, test_dimension - 1,
                                     batch_size, 1, cur->nb[1], cur->nb[2],
                                     cur->nb[3], (1) * sizeof(float) * 1024));

  ggml_tensor *temp_language_model_head_layer_norm_weights =
      ggml_repeat(ctx0, model.language_model_head_layer_norm_weights,
                  ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

  ggml_build_forward_expand(gf, temp_language_model_head_layer_norm_weights);

  cur = ggml_mul(ctx0, cur, temp_language_model_head_layer_norm_weights);
  cur = ggml_add(ctx0, cur, model.language_model_head_layer_norm_bias);

  ggml_tensor *final_output =
      ggml_cont(ctx0, ggml_view_4d(ctx0, cur, 1024, token_count, batch_size, 1,
                                   cur->nb[1], cur->nb[2], cur->nb[3],
                                   (1) * sizeof(float) * 1024));

  // std::cout << "mel token count: " << mel_token_count << std::endl;

  ggml_tensor *final_output_2 = ggml_cont(
      ctx0, ggml_view_4d(ctx0, cur, 1024, (mel_token_count / batch_size) - 2,
                         batch_size, 1, cur->nb[1], cur->nb[2], cur->nb[3],
                         (token_count) * sizeof(float) * 1024));

  ggml_set_name(final_output_2, "cur");

  ggml_build_forward_expand(gf, final_output_2);

  ggml_free(ctx0);
  return gf;
}

// clang-format off


/*
 
  ██████╗ ██████╗ ████████╗   ██████╗     
 ██╔════╝ ██╔══██╗╚══██╔══╝   ╚════██╗    
 ██║  ███╗██████╔╝   ██║█████╗ █████╔╝    
 ██║   ██║██╔═══╝    ██║╚════╝██╔═══╝     
 ╚██████╔╝██║        ██║      ███████╗    
  ╚═════╝ ╚═╝        ╚═╝      ╚══════╝    
                                          
  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
 ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
 ██║  ███╗██████╔╝███████║██████╔╝███████║
 ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
 ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
                                          
 
*/

// clang-format on

struct ggml_cgraph *
autoregressive_graph(const autoregressive_model &model,
                     const std::vector<int> mel_transformer_inputs_vector,
                     const std::vector<gpt_vocab::id> &tokens,
                     const bool fake_inputs, const int n_past,
                     const int fixed_position, const int batch_size) {

  const int token_count = tokens.size();

  static size_t buf_size = ggml_tensor_overhead() * GPT2_MAX_NODES +
                           ggml_graph_overhead_custom(GPT2_MAX_NODES, false);
  static std::vector<uint8_t> buf(buf_size);

  struct ggml_init_params params = {
      /*.mem_size   =*/buf_size,
      /*.mem_buffer =*/buf.data(),
      /*.no_alloc   =*/true, // the tensors will be allocated later by
                             // ggml_gallocr_alloc_graph()
  };

  struct ggml_context *ctx0 = ggml_init(params);

  struct ggml_cgraph *gf = ggml_new_graph_custom(ctx0, GPT2_MAX_NODES, false);

  struct ggml_tensor *input =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_count);

  ggml_set_name(input, "input_tokens");

  struct ggml_tensor *position =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_count);

  ggml_set_name(position, "input_position");

  struct ggml_tensor *conditioning_latent =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1024);

  ggml_set_name(conditioning_latent, "auto_conditioning");

  ggml_tensor *gpt2_input;

  if (fake_inputs) // 4 corresponds to batch of 4 sequences each with length 1
  {

    struct ggml_tensor *text_embedding =
        ggml_get_rows(ctx0, model.text_embedding_weights, input);
    struct ggml_tensor *text_position_embedding =
        ggml_get_rows(ctx0, model.text_position_embedding_weights, position);

    struct ggml_tensor *reshaped_latent =
        ggml_reshape_4d(ctx0, conditioning_latent, 1024, 1, 1, 1);

    struct ggml_tensor *embedding =
        ggml_add(ctx0, text_embedding, text_position_embedding);

    struct ggml_tensor *reshaped_embedding =
        ggml_reshape_4d(ctx0, embedding, 1024, 1, token_count, 1);

    struct ggml_tensor *mel_transformer_inputs = ggml_new_tensor_1d(
        ctx0, GGML_TYPE_I32, mel_transformer_inputs_vector.size());

    ggml_set_name(mel_transformer_inputs, "input_mel_tokens");

    mel_transformer_inputs =
        ggml_reshape_2d(ctx0, mel_transformer_inputs, batch_size,
                        mel_transformer_inputs_vector.size() / batch_size);

    ggml_build_forward_expand(gf, mel_transformer_inputs);

    struct ggml_tensor *truncated_mel_transformer_inputs =
        ggml_new_tensor_1d(ctx0, GGML_TYPE_I32,
                           batch_size); // hardcoding this instead of slicing it
                                        // from mel_transformer_inputs

    ggml_set_name(truncated_mel_transformer_inputs,
                  "input_mel_tokens_truncated");

    struct ggml_tensor *mel_embedding = ggml_get_rows(
        ctx0, model.mel_embedding_weights, truncated_mel_transformer_inputs);

    struct ggml_tensor *mel_position =
        ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);

    ggml_set_name(mel_position, "input_mel_position");

    struct ggml_tensor *mel_position_embedding =
        ggml_get_rows(ctx0, model.mel_position_embedding_weights, mel_position);

    mel_embedding = ggml_add(ctx0, mel_embedding, mel_position_embedding);

    ggml_set_name(mel_embedding, "mel_embedding");

    struct ggml_tensor *output =
        ggml_concat(ctx0, reshaped_latent, reshaped_embedding, 2);

    struct ggml_tensor *repeated_output =
        ggml_new_tensor_1d(ctx0, GGML_TYPE_F32,
                           batch_size * (tokens.size() + 1) *
                               1024); // todo do this more cleanly, going to
                                      // rely on 1d copy for same of simplicity
    output = ggml_reshape_1d(ctx0, output, (tokens.size() + 1) * 1024);

    repeated_output = ggml_repeat(ctx0, output, repeated_output);
    repeated_output = ggml_reshape_4d(ctx0, repeated_output, 1024,
                                      (tokens.size() + 1), batch_size, 1);

    repeated_output =
        ggml_cont(ctx0, ggml_permute(ctx0, repeated_output, 0, 2, 1, 3));

    ggml_set_name(repeated_output, "repeated_output");

    gpt2_input = ggml_concat(ctx0, repeated_output, mel_embedding, 2);

    gpt2_input = ggml_permute(ctx0, gpt2_input, 0, 2, 1, 3);

    ggml_tensor *final_latent =
        ggml_cpy(ctx0, gpt2_input,
                 ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, gpt2_input->ne));

    ggml_set_name(final_latent, "gpt2 input");
    ggml_build_forward_expand(gf, final_latent);

  } else {
    struct ggml_tensor *mel_transformer_inputs =
        ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);

    ggml_set_name(mel_transformer_inputs, "input_mel_tokens");

    mel_transformer_inputs =
        ggml_reshape_2d(ctx0, mel_transformer_inputs, batch_size, 1);

    struct ggml_tensor *mel_embedding = ggml_get_rows(
        ctx0, model.mel_embedding_weights, mel_transformer_inputs);
    ggml_set_name(mel_embedding, "mel embedding");

    struct ggml_tensor *fixed_embedding_ids =
        ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);

    ggml_set_name(fixed_embedding_ids, "input_fixed_embedding_ids");

    ggml_tensor *fixed_embedding = ggml_get_rows(
        ctx0, model.mel_position_embedding_weights, fixed_embedding_ids);

    ggml_set_name(fixed_embedding, "input_fixed embedding");

    gpt2_input = ggml_add(ctx0, mel_embedding, fixed_embedding);
    gpt2_input = ggml_cont(ctx0, ggml_permute(ctx0, gpt2_input, 0, 2, 1, 3));
    ggml_set_name(gpt2_input, "gpt2 input");
  }

  int test_dimension = gpt2_input->ne[1];

  struct ggml_tensor *cur = ggml_cont(ctx0, gpt2_input);

  struct ggml_tensor *Qcur;
  struct ggml_tensor *Kcur;
  struct ggml_tensor *Vcur;

  struct ggml_tensor *Q;
  struct ggml_tensor *K;

  struct ggml_tensor *KQ;
  struct ggml_tensor *KQ_scaled;
  struct ggml_tensor *KQ_masked;
  struct ggml_tensor *KQ_soft_max;
  struct ggml_tensor *V_transposed;
  struct ggml_tensor *KQV;
  struct ggml_tensor *KQV_merged;

  struct ggml_tensor *residual;
  struct ggml_tensor *feed_forward_residual;

  struct ggml_tensor *test;
  for (int i = 0; i < 30; i++) {

    residual = ggml_cpy(ctx0, cur,
                        ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1024,
                                           test_dimension, batch_size, 1));

    // ggml_build_forward_expand(gf, residual);
    // layer norm

    cur = ggml_norm(ctx0, cur, 1e-05);

    ggml_tensor *temp_cur =
        ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

    ggml_set_name(temp_cur, "postnorm");
    ggml_build_forward_expand(gf, temp_cur);

    ggml_format_name(cur, "l%d.norm", i);

    ggml_tensor *temp_ln_1_weights =
        ggml_repeat(ctx0, model.layers[i].layer_norm_1_weights,
                    ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

    ggml_set_name(temp_ln_1_weights, "weights");
    // ggml_build_forward_expand(gf, temp_ln_1_weights);

    cur = ggml_mul(
        ctx0, cur,
        temp_ln_1_weights); // if you flip the order of this it doesn't work on
                            // the second token generation process.TODO why does
                            // flipping the order of this break it?

    cur = ggml_add(ctx0, cur, model.layers[i].layer_norm_1_bias);

    // ggml_tensor * temp_weights = ggml_cpy(ctx0,
    // model.layers[i].layer_norm_1_weights, ggml_new_tensor(ctx0,
    // GGML_TYPE_F32,4,model.layers[i].layer_norm_1_weights->ne) );
    ggml_tensor *temp_bias =
        ggml_cpy(ctx0, model.layers[i].layer_norm_1_bias,
                 ggml_new_tensor(ctx0, GGML_TYPE_F32, 4,
                                 model.layers[i].layer_norm_1_bias->ne));

    //  if(!fake_inputs)
    // {
    // ggml_build_forward_expand(gf, temp_bias);
    // ggml_set_name(temp_bias, "bias");
    // return gf;
    // }

    // this is implemented as conv1d in pytorch, but it's actually just a affine
    // transformation with a weight and bias
    cur = ggml_mul_mat(
        ctx0,
        ggml_reshape_2d(
            ctx0,
            ggml_cont(ctx0,
                      ggml_transpose(
                          ctx0, model.layers[i].c_attention_attention_weights)),
            1024, 3072),
        cur);

    cur = ggml_reshape_4d(ctx0, cur, 3072, test_dimension, batch_size, 1);

    cur = ggml_add(
        ctx0, cur,
        model.layers[i]
            .c_attention_attention_bias); // worth studying the diffs here, why
                                          // did I have to remove this repeat
                                          // for settings where the second
                                          // dimension is 1?

    // derived from ggml reference gpt-2 implementation
    Qcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, 1024, test_dimension,
                                        batch_size, cur->nb[1], cur->nb[2], 0));

    // Kcur = ggml_cont(ctx0,ggml_permute(ctx0,ggml_view_4d(ctx0, cur, 1024,
    // test_dimension, 4, 1,cur->nb[1], cur->nb[2],cur->nb[3], 1024 *
    // sizeof(float)),0,1,3,2));

    Kcur = ggml_cont(
        ctx0,
        ggml_permute(ctx0,
                     ggml_view_3d(ctx0, cur, 1024, test_dimension, batch_size,
                                  cur->nb[1], cur->nb[2], 1024 * sizeof(float)),
                     0, 2, 1, 3));
    Vcur = ggml_cont(
        ctx0,
        ggml_permute(ctx0,
                     ggml_view_3d(ctx0, cur, 1024, test_dimension, batch_size,
                                  cur->nb[1], cur->nb[2], 2048 * sizeof(float)),
                     0, 2, 1, 3));

    struct ggml_tensor *k = ggml_view_1d(
        ctx0, model.memory_key, 1024 * test_dimension * batch_size,
        (ggml_element_size(model.memory_key) *
         ((i * 404 * 1024 * batch_size) + ((n_past)*1024 * batch_size))));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));

    struct ggml_tensor *v = ggml_view_1d(
        ctx0, model.memory_value, 1024 * test_dimension * batch_size,
        (ggml_element_size(model.memory_value) *
         ((i * 404 * 1024 * batch_size) + ((n_past)*1024 * batch_size))));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));

    // num heads 16
    // head dim 64

    Q = ggml_cont(ctx0,
                  ggml_permute(ctx0,
                               ggml_reshape_4d(ctx0, Qcur, 64, 16,
                                               test_dimension, batch_size),
                               0, 2, 1, 3));

    // this is likely not general and but should work for the first layer, may
    // need generalizing once we reach
    // the end of the first layer.
    K = ggml_cont(
        ctx0,
        ggml_permute(
            ctx0,
            ggml_reshape_4d(
                ctx0,
                ggml_view_1d(ctx0, model.memory_key,
                             1024 * (test_dimension + n_past) * batch_size,
                             (ggml_element_size(model.memory_key) *
                              (i * 404 * 1024 * batch_size))),
                64, 16, batch_size,
                test_dimension + n_past), // flagged possible batch size
            0, 2, 3, 1));

    V_transposed = ggml_cont(
        ctx0,
        ggml_permute(
            ctx0,
            ggml_reshape_4d(
                ctx0,
                ggml_view_1d(ctx0, model.memory_value,
                             1024 * (test_dimension + n_past) * batch_size,
                             (ggml_element_size(model.memory_value) *
                              (i * 404 * 1024 * batch_size))),
                64, 16, batch_size, test_dimension + n_past),
            1, 2, 3, 0));

    KQ = ggml_mul_mat(ctx0, K, Q);

    KQ_scaled = ggml_scale_inplace(ctx0, KQ, 1.0f / sqrt(float(64)));

    KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

    KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

    KQV = ggml_mul_mat(ctx0, KQ_soft_max, V_transposed);

    ggml_set_name(KQ_soft_max, "after KQ");
    ggml_set_name(V_transposed, "after V");

    // getting the initial KQV value
    KQV = ggml_reshape_3d(ctx0, KQV, test_dimension, 64, 16 * batch_size);
    KQV = ggml_permute(ctx0, KQV, 1, 0, 2, 3);
    KQV = ggml_cont_3d(ctx0, KQV, 64, test_dimension, 16 * batch_size);
    KQV = ggml_reshape_4d(ctx0, KQV, 64, test_dimension, 16, batch_size);

    //"merge heads" operation
    KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
    KQV_merged =
        ggml_cont_3d(ctx0, KQV_merged, 1024, test_dimension, batch_size);

    cur = ggml_mul_mat(
        ctx0,
        ggml_reshape_2d(
            ctx0,
            ggml_cont(
                ctx0,
                ggml_transpose(ctx0,
                               model.layers[i].c_attention_projection_weights)),
            1024, 1024),
        KQV_merged);

    cur = ggml_add(ctx0, cur, model.layers[i].c_attention_projection_bias);

    // layer input passthrough
    // cur = ggml_add(ctx0, cur,residual);

    cur = ggml_add(
        ctx0, ggml_reshape_1d(ctx0, cur, 1024 * batch_size * test_dimension),
        ggml_reshape_1d(
            ctx0, residual,
            1024 * batch_size *
                test_dimension)); // it's really strange that this is necessary,
                                  // why does it have different behavior than
                                  // commented out above
    cur = ggml_reshape_4d(ctx0, cur, 1024, test_dimension, batch_size, 1);

    feed_forward_residual =
        ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

    // layer norm 2
    cur = ggml_norm(ctx0, cur, 1e-05);

    ggml_format_name(cur, "l%d.norm_2", i);

    ggml_tensor *temp_ln_2_weights =
        ggml_repeat(ctx0, model.layers[i].layer_norm_2_weights,
                    ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

    ggml_set_name(temp_ln_2_weights, "test");

    ggml_build_forward_expand(gf, temp_ln_2_weights);

    cur = ggml_mul(ctx0, cur, temp_ln_2_weights);
    cur = ggml_add(ctx0, cur, model.layers[i].layer_norm_2_bias);

    //  fully connected multi layer perceptron
    cur = ggml_mul_mat(
        ctx0,
        ggml_reshape_2d(
            ctx0,
            ggml_cont(
                ctx0,
                ggml_transpose(
                    ctx0,
                    model.layers[i]
                        .c_multi_layer_perceptron_fully_connected_weights)),
            1024, 4096),
        cur);

    cur =
        ggml_add(ctx0, cur,
                 model.layers[i].c_multi_layer_perceptron_fully_connected_bias);

    // gelu
    cur = ggml_gelu(ctx0, cur);

    // mlp fully connected
    cur = ggml_mul_mat(
        ctx0,
        ggml_reshape_2d(
            ctx0,
            ggml_cont(
                ctx0,
                ggml_transpose(
                    ctx0, model.layers[i]
                              .c_multi_layer_perceptron_projection_weights)),
            4096, 1024),
        cur);

    cur = ggml_add(ctx0, cur,
                   model.layers[i].c_multi_layer_perceptron_projection_bias);

    // final residual
    // another case where I had to flatten before adding to get correct results.
    // Either ggml had a pre-existing bug for batch addition, or one of my
    // modifications introduced it. This will need to be addressed.

    cur = ggml_add(
        ctx0,
        ggml_reshape_1d(ctx0, cur, 1024 * test_dimension * batch_size * 1),
        ggml_reshape_1d(ctx0, feed_forward_residual,
                        1024 * test_dimension * batch_size * 1));

    cur = ggml_reshape_4d(ctx0, cur, 1024, test_dimension, batch_size, 1);
    // cur = ggml_add(ctx0, cur, feed_forward_residual);
  }

  cur = ggml_norm(ctx0, cur, 1e-05);

  ggml_tensor *temp_final_layer_norm_weights =
      ggml_repeat(ctx0, model.final_layer_norm_weights,
                  ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

  ggml_build_forward_expand(gf, temp_final_layer_norm_weights);

  cur = ggml_mul(ctx0, cur, temp_final_layer_norm_weights);
  cur = ggml_add(ctx0, cur, model.final_layer_norm_bias);

  cur = ggml_norm(ctx0, cur, 1e-05);

  ggml_tensor *temp_language_model_head_layer_norm_weights =
      ggml_repeat(ctx0, model.language_model_head_layer_norm_weights,
                  ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

  cur = ggml_mul(ctx0, cur, temp_language_model_head_layer_norm_weights);
  cur = ggml_add(ctx0, cur, model.language_model_head_layer_norm_bias);

  cur = ggml_mul_mat(ctx0, model.language_model_head_linear_weights, cur);

  cur = ggml_add(ctx0, cur, model.language_model_head_linear_bias);

  ggml_tensor *next_token_logits = ggml_cont(
      ctx0, ggml_view_4d(ctx0, cur, 8194, 1, batch_size, 1, cur->nb[1],
                         cur->nb[2], cur->nb[3],
                         (test_dimension - 1) * sizeof(float) *
                             8194)); // this "test_dimension - 1" business
                                     // slices off the last batch of logits

  next_token_logits =
      ggml_reshape_4d(ctx0, next_token_logits, 8194, batch_size, 1, 1);

  ggml_set_name(next_token_logits, "next token logits");

  // mel_transformer_inputs = ggml_reshape_4d(ctx0, mel_transformer_inputs, 18,
  // 4, 1, 1);

  // ggml_tensor * score = ggml_gather(ctx0, next_token_logits,
  // mel_transformer_inputs, 1);

  // std::cout << "didn't reach here" << std::endl;

  ggml_build_forward_expand(gf, next_token_logits);

  // embd_w.resize(n_vocab);
  // memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)),
  // sizeof(float)*n_vocab);

  // std::cout << "reached end graph build" << std::endl;

  ggml_free(ctx0);

  return gf;
}

// clang-format off


/*
 
 ██████╗ ██╗███████╗███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗    
 ██╔══██╗██║██╔════╝██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║    
 ██║  ██║██║█████╗  █████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║    
 ██║  ██║██║██╔══╝  ██╔══╝  ██║   ██║╚════██║██║██║   ██║██║╚██╗██║    
 ██████╔╝██║██║     ██║     ╚██████╔╝███████║██║╚██████╔╝██║ ╚████║    
 ╚═════╝ ╚═╝╚═╝     ╚═╝      ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝    
                                                                       
  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗                             
 ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║                             
 ██║  ███╗██████╔╝███████║██████╔╝███████║                             
 ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║                             
 ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║                             
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝                             
                                                                       
 
*/

// clang-format on

struct ggml_cgraph *diffusion_graph(struct diffusion_model &model,
                                    std::vector<float> &latent,
                                    int output_sequence_length,
                                    int diffusion_index,
                                    bool conditioning_free) {
  static size_t buf_size = ggml_tensor_overhead() * GPT2_MAX_NODES +
                           ggml_graph_overhead_custom(GPT2_MAX_NODES, false);
  static std::vector<uint8_t> buf(buf_size);

  struct ggml_init_params params = {
      /*.mem_size   =*/buf_size,
      /*.mem_buffer =*/buf.data(),
      /*.no_alloc   =*/true, // the tensors will be allocated later by
                             // ggml_gallocr_alloc_graph()
  };

  int latent_length = latent.size() / 1024;

  int time_embedding_size = 80;

  struct ggml_context *ctx0 = ggml_init(params);

  struct ggml_cgraph *gf = ggml_new_graph_custom(ctx0, GPT2_MAX_NODES, false);

  std::vector<ggml_tensor *> time_embedding_tensors(time_embedding_size);
  for (int i = 0; i < time_embedding_size; i++) {
    time_embedding_tensors[i] = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1024);
    ggml_set_name(time_embedding_tensors[i],
                  ("time_embedding_" + std::to_string(i)).c_str());
    ggml_build_forward_expand(gf, time_embedding_tensors[i]);
  }

  struct ggml_tensor *latent_tensor =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, latent.size());

  ggml_set_name(latent_tensor, "input_latent_tensor");

  struct ggml_tensor *noise_tensor =
      ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, output_sequence_length, 100);

  ggml_set_name(noise_tensor, "noise_tensor");

  struct ggml_tensor *relative_position_buckets_tensor =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, latent_length * latent_length);

  ggml_set_name(relative_position_buckets_tensor,
                "relative_position_buckets_tensor");

  struct ggml_tensor *output_relative_position_buckets_tensor =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_I32,
                         output_sequence_length * output_sequence_length);

  ggml_set_name(output_relative_position_buckets_tensor,
                "output_relative_position_buckets_tensor");

  struct ggml_tensor *conditioning_scale_offset =
      ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

  ggml_set_name(conditioning_scale_offset, "conditioning_scale_offset");

  ggml_build_forward_expand(gf, latent_tensor);
  ggml_build_forward_expand(gf, relative_position_buckets_tensor);
  ggml_build_forward_expand(gf, output_relative_position_buckets_tensor);
  ggml_build_forward_expand(gf, conditioning_scale_offset);
  ggml_build_forward_expand(gf, noise_tensor);

  /*
  // avoid writing to tensors if we are only measuring the memory usage
  if (!ggml_allocr_is_measure(allocr)) {
      ggml_backend_tensor_set(latent_tensor, latent.data(), 0,
  latent.size()*ggml_element_size(latent_tensor));
  }
  */

  latent_tensor = ggml_reshape_3d(ctx0, latent_tensor, 1024, latent_length, 1);
  latent_tensor =
      ggml_cont(ctx0, ggml_permute(ctx0, latent_tensor, 1, 0, 2, 3));

  ggml_tensor *cur = latent_tensor;

  ggml_tensor *residual;

  ggml_tensor *Q;
  ggml_tensor *K;
  ggml_tensor *V;

  ggml_tensor *KQ;
  ggml_tensor *KQ_scaled;
  ggml_tensor *relative_position_bias_weights;

  if (!conditioning_free) {
    ggml_tensor *conditioning_scale =
        ggml_view_1d(ctx0, model.diffusion_conditioning_latent, 1024, 0);
    ggml_tensor *conditioning_shift = ggml_view_1d(
        ctx0, model.diffusion_conditioning_latent, 1024,
        ggml_element_size(model.diffusion_conditioning_latent) * 1024);

    cur = ggml_cont(
        ctx0, ggml_conv_1d(ctx0, model.latent_conditioner_convolution_weight,
                           cur, 1, 1, 1));

    cur = ggml_cont(
        ctx0,
        ggml_transpose(
            ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                           model.latent_conditioner_convolution_bias)));

    // latent conditioner attention blocks

    for (int i = 0; i < 4; i++) {

      // group norm

      residual =
          ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

      cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);

      cur = ggml_group_norm(ctx0, cur, 32, TORTOISE_NORM_EPSILON);

      cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1024);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      cur = ggml_mul(
          ctx0, cur,
          ggml_repeat(ctx0,
                      model.latent_conditioner_attention_blocks[i].norm_weight,
                      ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne)));

      cur = ggml_add(ctx0, cur,
                     model.latent_conditioner_attention_blocks[i].norm_bias);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      // qkv

      cur = ggml_cont(
          ctx0,
          ggml_conv_1d(
              ctx0,
              ggml_reshape_3d(
                  ctx0, model.latent_conditioner_attention_blocks[i].qkv_weight,
                  1, 1024, 3072),
              cur, 1, 0, 1));

      // cur =
      //     ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4,
      //     cur->ne));

      cur = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0,
              ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                       model.latent_conditioner_attention_blocks[i].qkv_bias)));

      cur = ggml_reshape_3d(ctx0, cur, latent_length, 192, 16);

      // derived from ggml reference gpt-2 implementation
      Q = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, latent_length, 64, 16,
                                       cur->nb[1], cur->nb[2], 0));

      K = ggml_cont(ctx0,
                    ggml_view_3d(ctx0, cur, latent_length, 64, 16, cur->nb[1],
                                 cur->nb[2],
                                 latent_length * 64 * ggml_element_size(cur)));

      V = ggml_cont(ctx0,
                    ggml_view_3d(ctx0, cur, latent_length, 64, 16, cur->nb[1],
                                 cur->nb[2],
                                 latent_length * 128 * ggml_element_size(cur)));

      KQ = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, K)),
                        ggml_cont(ctx0, ggml_transpose(ctx0, Q)));

      KQ_scaled = ggml_scale_inplace(ctx0, KQ, 1.0f / sqrt(float(64)));

      relative_position_bias_weights = ggml_reshape_3d(
          ctx0,
          ggml_get_rows(
              ctx0,
              model.latent_conditioner_attention_blocks[i]
                  .relative_position_embeddings_relative_attention_bias_weight,
              relative_position_buckets_tensor),
          16, latent_length, latent_length);

      relative_position_bias_weights = ggml_cont(
          ctx0, ggml_permute(ctx0, relative_position_bias_weights, 2, 0, 1, 3));

      relative_position_bias_weights =
          ggml_scale_inplace(ctx0, relative_position_bias_weights, 8.0);

      cur = ggml_add(ctx0, relative_position_bias_weights, KQ_scaled);

      cur = ggml_soft_max_inplace(ctx0, cur);

      cur = ggml_cont(
          ctx0, (ggml_transpose(
                    ctx0, ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, cur, V),
                                          latent_length, 1024))));

      cur = ggml_mul_mat(
          ctx0,
          model.latent_conditioner_attention_blocks[i].projection_out_weight,
          cur);

      cur = ggml_cont(
          ctx0, ggml_transpose(
                    ctx0, ggml_add(ctx0, cur,
                                   model.latent_conditioner_attention_blocks[i]
                                       .projection_out_bias)));

      cur = ggml_add(ctx0, cur, residual);
    }

    // code norm (group norm)

    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);

    cur = ggml_group_norm(ctx0, cur, 32, TORTOISE_NORM_EPSILON);

    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1024);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    cur =
        ggml_mul(ctx0, cur,
                 ggml_repeat(ctx0, model.code_norm_weight,
                             ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne)));

    cur = ggml_add(ctx0, cur, model.code_norm_bias);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    ggml_tensor *offset_conditioning_scale =
        ggml_add(ctx0, conditioning_scale, conditioning_scale_offset);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    cur = ggml_mul(ctx0, cur, offset_conditioning_scale);

    cur = ggml_add(ctx0, cur, conditioning_shift);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    cur = ggml_upscale_ext(ctx0, cur, output_sequence_length, 1024, 1, 1);
  } else {
    cur = ggml_cont(
        ctx0,
        ggml_transpose(
            ctx0, ggml_repeat(ctx0, model.unconditioned_embedding,
                              ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1024,
                                                 output_sequence_length))));
  }

  ggml_tensor *emb = time_embedding_tensors[diffusion_index];

  emb = ggml_mul_mat(ctx0, model.time_emb_linear_0_weight, emb);

  emb = ggml_add(ctx0, emb, model.time_emb_linear_0_bias);

  emb = ggml_silu(ctx0, emb);

  emb = ggml_mul_mat(ctx0, model.time_emb_linear_1_weight, emb);

  emb = ggml_add(ctx0, emb, model.time_emb_layer_norm_1_bias);

  ggml_tensor *time_embedding = emb;

  ggml_tensor *code_embedding;

  for (int c = 0; c < 3; c++) {

    // Resblock

    {
      ggml_tensor *pass_through = cur;

      // in_layers

      cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);

      cur = ggml_group_norm(ctx0, cur, 32, TORTOISE_NORM_EPSILON);

      cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1024);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      cur = ggml_mul(
          ctx0, cur,
          ggml_repeat(ctx0,
                      model.timestep_conditioning_integrator_diffusion_layers[c]
                          .resblock_in_layers_0_weight,
                      ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne)));

      cur = ggml_add(ctx0, cur,
                     model.timestep_conditioning_integrator_diffusion_layers[c]
                         .resblock_in_layers_0_bias);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      cur = ggml_silu(ctx0, cur);

      cur = ggml_cont(
          ctx0,
          ggml_conv_1d(
              ctx0,
              ggml_reshape_3d(
                  ctx0,
                  model.timestep_conditioning_integrator_diffusion_layers[c]
                      .resblock_in_layers_2_weight,
                  1, 1024, 1024),
              cur, 1, 0, 1));

      cur = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0,
              ggml_add(
                  ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                  model.timestep_conditioning_integrator_diffusion_layers[c]
                      .resblock_in_layers_2_bias)));

      // emb_layers

      ggml_tensor *temp_emb = ggml_silu(ctx0, emb);

      temp_emb = ggml_mul_mat(
          ctx0, temp_emb,
          model.timestep_conditioning_integrator_diffusion_layers[c]
              .resblock_emb_layers_1_weight);

      temp_emb = ggml_cont(ctx0, ggml_transpose(ctx0, temp_emb));

      temp_emb =
          ggml_add(ctx0, temp_emb,
                   model.timestep_conditioning_integrator_diffusion_layers[c]
                       .resblock_emb_layers_1_bias);

      // out layers

      ggml_tensor *emb_scale = ggml_view_1d(ctx0, temp_emb, 1024, 0);
      ggml_tensor *emb_shift = ggml_view_1d(ctx0, temp_emb, 1024,
                                            ggml_element_size(temp_emb) * 1024);

      cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);

      cur = ggml_group_norm(ctx0, cur, 32, TORTOISE_NORM_EPSILON);

      cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1024);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      cur = ggml_mul(
          ctx0, cur,
          ggml_repeat(ctx0,
                      model.timestep_conditioning_integrator_diffusion_layers[c]
                          .resblock_out_layers_0_weight,
                      ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne)));

      cur = ggml_add(ctx0, cur,
                     model.timestep_conditioning_integrator_diffusion_layers[c]
                         .resblock_out_layers_0_bias);

      cur = ggml_mul(ctx0, cur,
                     ggml_add(ctx0, emb_scale, conditioning_scale_offset));

      cur = ggml_add(ctx0, cur, emb_shift);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      cur = ggml_silu(ctx0, cur);

      cur = ggml_cont(
          ctx0,
          ggml_conv_1d(
              ctx0,
              ggml_reshape_3d(
                  ctx0,
                  model.timestep_conditioning_integrator_diffusion_layers[c]
                      .resblock_out_layers_3_weight,
                  3, 1024, 1024),
              cur, 1, 1, 1));

      cur = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0,
              ggml_add(
                  ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                  model.timestep_conditioning_integrator_diffusion_layers[c]
                      .resblock_out_layers_3_bias)));

      cur =
          ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

      cur = ggml_add(ctx0, cur, pass_through);
    }

    // Attention block
    {
      // group norm

      residual =
          ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

      cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);

      cur = ggml_group_norm(ctx0, cur, 32, TORTOISE_NORM_EPSILON);

      cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1024);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      cur = ggml_mul(
          ctx0, cur,
          ggml_repeat(ctx0,
                      model.timestep_conditioning_integrator_diffusion_layers[c]
                          .attn_norm_weight,
                      ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne)));

      cur = ggml_add(ctx0, cur,
                     model.timestep_conditioning_integrator_diffusion_layers[c]
                         .attn_norm_bias);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      // qkv

      cur = ggml_cont(
          ctx0,
          ggml_conv_1d(
              ctx0,
              ggml_reshape_3d(
                  ctx0,
                  model.timestep_conditioning_integrator_diffusion_layers[c]
                      .attn_qkv_weight,
                  1, 1024, 3072),
              cur, 1, 0, 1));

      cur = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0,
              ggml_add(
                  ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                  model.timestep_conditioning_integrator_diffusion_layers[c]
                      .attn_qkv_bias)));

      cur = ggml_reshape_3d(ctx0, cur, output_sequence_length, 192, 16);

      // derived from ggml reference gpt-2 implementation
      Q = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, output_sequence_length, 64,
                                       16, cur->nb[1], cur->nb[2], 0));

      K = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, output_sequence_length, 64,
                                       16, cur->nb[1], cur->nb[2],
                                       output_sequence_length * 64 *
                                           ggml_element_size(cur)));

      V = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, output_sequence_length, 64,
                                       16, cur->nb[1], cur->nb[2],
                                       output_sequence_length * 128 *
                                           ggml_element_size(cur)));

      KQ = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, K)),
                        ggml_cont(ctx0, ggml_transpose(ctx0, Q)));

      KQ_scaled = ggml_scale_inplace(ctx0, KQ, 1.0f / sqrt(float(64)));

      relative_position_bias_weights = ggml_reshape_3d(
          ctx0,
          ggml_get_rows(
              ctx0,
              model.timestep_conditioning_integrator_diffusion_layers[c]
                  .attn_relative_pos_embeddings_relative_attention_bias_weight,
              output_relative_position_buckets_tensor),
          16, output_sequence_length, output_sequence_length);

      relative_position_bias_weights = ggml_cont(
          ctx0, ggml_permute(ctx0, relative_position_bias_weights, 2, 0, 1, 3));

      relative_position_bias_weights =
          ggml_scale_inplace(ctx0, relative_position_bias_weights, 8.0);

      cur = ggml_add(ctx0, relative_position_bias_weights, KQ_scaled);

      cur = ggml_soft_max_inplace(ctx0, cur);

      cur = ggml_cont(
          ctx0, (ggml_transpose(
                    ctx0, ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, cur, V),
                                          output_sequence_length, 1024))));

      cur = ggml_mul_mat(
          ctx0,
          model.timestep_conditioning_integrator_diffusion_layers[c]
              .attn_proj_out_weight,
          cur);

      cur = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0,
              ggml_add(
                  ctx0, cur,
                  model.timestep_conditioning_integrator_diffusion_layers[c]
                      .attn_proj_out_bias)));

      cur = ggml_add(ctx0, cur, residual);
    }
  }

  code_embedding = cur;

  cur = noise_tensor;

  cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

  cur = ggml_cont(
      ctx0,
      ggml_conv_1d(ctx0,
                   ggml_reshape_3d(ctx0, model.inp_block_weight, 3, 100, 1024),
                   cur, 1, 1, 1));

  cur = ggml_cont(
      ctx0, ggml_transpose(
                ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                               model.inp_block_bias)));

  cur = ggml_reshape_2d(ctx0, ggml_concat(ctx0, cur, code_embedding, 2),
                        output_sequence_length, 2048);

  cur = ggml_cont(
      ctx0, ggml_conv_1d(ctx0,
                         ggml_reshape_3d(ctx0, model.integrating_conv_weight, 1,
                                         2048, 1024),
                         cur, 1, 0, 1));

  cur = ggml_cont(
      ctx0, ggml_transpose(
                ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                               model.integrating_conv_bias)));

  cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

  for (int c = 0; c < 10; c++) {
    // Resblock

    {
      ggml_tensor *pass_through = cur;

      // in_layers

      cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);

      cur = ggml_group_norm(ctx0, cur, 32, TORTOISE_NORM_EPSILON);

      cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1024);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      cur = ggml_mul(
          ctx0, cur,
          ggml_repeat(
              ctx0, model.main_diffusion_layers[c].resblock_in_layers_0_weight,
              ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne)));

      cur = ggml_add(ctx0, cur,
                     model.main_diffusion_layers[c].resblock_in_layers_0_bias);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      cur = ggml_silu(ctx0, cur);

      cur = ggml_cont(
          ctx0, ggml_conv_1d(ctx0,
                             ggml_reshape_3d(ctx0,
                                             model.main_diffusion_layers[c]
                                                 .resblock_in_layers_2_weight,
                                             1, 1024, 1024),
                             cur, 1, 0, 1));

      cur = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0,
              ggml_add(
                  ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                  model.main_diffusion_layers[c].resblock_in_layers_2_bias)));

      cur =
          ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

      // emb_layers

      ggml_tensor *temp_emb = ggml_silu(ctx0, time_embedding);

      temp_emb = ggml_mul_mat(
          ctx0, temp_emb,
          model.main_diffusion_layers[c].resblock_emb_layers_1_weight);

      temp_emb = ggml_cont(ctx0, ggml_transpose(ctx0, temp_emb));

      temp_emb =
          ggml_add(ctx0, temp_emb,
                   model.main_diffusion_layers[c].resblock_emb_layers_1_bias);

      // out layers

      ggml_tensor *emb_scale = ggml_view_1d(ctx0, temp_emb, 1024, 0);
      ggml_tensor *emb_shift = ggml_view_1d(ctx0, temp_emb, 1024,
                                            ggml_element_size(temp_emb) * 1024);

      cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);

      cur = ggml_group_norm(ctx0, cur, 32, TORTOISE_NORM_EPSILON);

      cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1024);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      cur = ggml_mul(
          ctx0, cur,
          ggml_repeat(
              ctx0, model.main_diffusion_layers[c].resblock_out_layers_0_weight,
              ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne)));

      cur = ggml_add(ctx0, cur,
                     model.main_diffusion_layers[c].resblock_out_layers_0_bias);

      cur = ggml_mul(ctx0, cur,
                     ggml_add(ctx0, emb_scale, conditioning_scale_offset));

      cur = ggml_add(ctx0, cur, emb_shift);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      cur = ggml_silu(ctx0, cur);

      cur = ggml_cont(
          ctx0, ggml_conv_1d(ctx0,
                             ggml_reshape_3d(ctx0,
                                             model.main_diffusion_layers[c]
                                                 .resblock_out_layers_3_weight,
                                             3, 1024, 1024),
                             cur, 1, 1, 1));

      cur = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0,
              ggml_add(
                  ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                  model.main_diffusion_layers[c].resblock_out_layers_3_bias)));

      cur =
          ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

      cur = ggml_add(ctx0, cur, pass_through);
    }

    // Attention block
    {
      // group norm

      residual =
          ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

      cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);

      cur = ggml_group_norm(ctx0, cur, 32, TORTOISE_NORM_EPSILON);

      cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1024);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      cur = ggml_mul(
          ctx0, cur,
          ggml_repeat(ctx0, model.main_diffusion_layers[c].attn_norm_weight,
                      ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne)));

      cur = ggml_add(ctx0, cur, model.main_diffusion_layers[c].attn_norm_bias);

      cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      // qkv

      cur = ggml_cont(
          ctx0,
          ggml_conv_1d(ctx0,
                       ggml_reshape_3d(
                           ctx0, model.main_diffusion_layers[c].attn_qkv_weight,
                           1, 1024, 3072),
                       cur, 1, 0, 1));

      cur = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                             model.main_diffusion_layers[c].attn_qkv_bias)));

      cur = ggml_reshape_3d(ctx0, cur, output_sequence_length, 192, 16);

      // derived from ggml reference gpt-2 implementation
      Q = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, output_sequence_length, 64,
                                       16, cur->nb[1], cur->nb[2], 0));

      K = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, output_sequence_length, 64,
                                       16, cur->nb[1], cur->nb[2],
                                       output_sequence_length * 64 *
                                           ggml_element_size(cur)));

      V = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, output_sequence_length, 64,
                                       16, cur->nb[1], cur->nb[2],
                                       output_sequence_length * 128 *
                                           ggml_element_size(cur)));

      KQ = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, K)),
                        ggml_cont(ctx0, ggml_transpose(ctx0, Q)));

      KQ_scaled = ggml_scale_inplace(ctx0, KQ, 1.0f / sqrt(float(64)));

      relative_position_bias_weights = ggml_reshape_3d(
          ctx0,
          ggml_get_rows(
              ctx0,
              model.main_diffusion_layers[c]
                  .attn_relative_pos_embeddings_relative_attention_bias_weight,
              output_relative_position_buckets_tensor),
          16, output_sequence_length, output_sequence_length);

      relative_position_bias_weights = ggml_cont(
          ctx0, ggml_permute(ctx0, relative_position_bias_weights, 2, 0, 1, 3));

      relative_position_bias_weights =
          ggml_scale_inplace(ctx0, relative_position_bias_weights, 8.0);

      cur = ggml_add(ctx0, relative_position_bias_weights, KQ_scaled);

      cur = ggml_soft_max_inplace(ctx0, cur);

      cur = ggml_cont(
          ctx0, (ggml_transpose(
                    ctx0, ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, cur, V),
                                          output_sequence_length, 1024))));

      cur = ggml_mul_mat(
          ctx0, model.main_diffusion_layers[c].attn_proj_out_weight, cur);

      cur = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0,
              ggml_add(ctx0, cur,
                       model.main_diffusion_layers[c].attn_proj_out_bias)));

      cur = ggml_add(ctx0, cur, residual);
    }
  }

  // layers 10-12 of main layers
  for (int c = 0; c < 3; c++) {
    // Resblock

    ggml_tensor *pass_through = cur;

    // in_layers

    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);

    cur = ggml_group_norm(ctx0, cur, 32, TORTOISE_NORM_EPSILON);

    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1024);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    cur = ggml_mul(
        ctx0, cur,
        ggml_repeat(ctx0, model.main_residual_blocks[c].in_layers_0_weight,
                    ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne)));

    cur = ggml_add(ctx0, cur, model.main_residual_blocks[c].in_layers_0_bias);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    cur = ggml_silu(ctx0, cur);

    cur = ggml_cont(
        ctx0,
        ggml_conv_1d(ctx0,
                     ggml_reshape_3d(
                         ctx0, model.main_residual_blocks[c].in_layers_2_weight,
                         1, 1024, 1024),
                     cur, 1, 0, 1));

    cur = ggml_cont(
        ctx0,
        ggml_transpose(
            ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                           model.main_residual_blocks[c].in_layers_2_bias)));

    // emb_layers

    ggml_tensor *temp_emb = ggml_silu(ctx0, time_embedding);

    temp_emb = ggml_mul_mat(ctx0, temp_emb,
                            model.main_residual_blocks[c].emb_layers_1_weight);

    temp_emb = ggml_cont(ctx0, ggml_transpose(ctx0, temp_emb));

    temp_emb = ggml_add(ctx0, temp_emb,
                        model.main_residual_blocks[c].emb_layers_1_bias);

    // out layers

    ggml_tensor *emb_scale = ggml_view_1d(ctx0, temp_emb, 1024, 0);
    ggml_tensor *emb_shift =
        ggml_view_1d(ctx0, temp_emb, 1024, ggml_element_size(temp_emb) * 1024);

    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);

    cur = ggml_group_norm(ctx0, cur, 32, TORTOISE_NORM_EPSILON);

    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1024);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    cur = ggml_mul(
        ctx0, cur,
        ggml_repeat(ctx0, model.main_residual_blocks[c].out_layers_0_weight,
                    ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne)));

    cur = ggml_add(ctx0, cur, model.main_residual_blocks[c].out_layers_0_bias);

    cur = ggml_mul(ctx0, cur,
                   ggml_add(ctx0, emb_scale, conditioning_scale_offset));

    cur = ggml_add(ctx0, cur, emb_shift);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    cur = ggml_silu(ctx0, cur);

    cur = ggml_cont(
        ctx0, ggml_conv_1d(
                  ctx0,
                  ggml_reshape_3d(
                      ctx0, model.main_residual_blocks[c].out_layers_3_weight,
                      3, 1024, 1024),
                  cur, 1, 1, 1));

    cur = ggml_cont(
        ctx0,
        ggml_transpose(
            ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                           model.main_residual_blocks[c].out_layers_3_bias)));

    cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

    cur = ggml_add(ctx0, cur, pass_through);
  }

  cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);

  cur = ggml_group_norm(ctx0, cur, 32, TORTOISE_NORM_EPSILON);

  cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1024);

  cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

  cur = ggml_mul(ctx0, cur,
                 ggml_repeat(ctx0, model.out_group_norm_weight,
                             ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne)));

  cur = ggml_add(ctx0, cur, model.out_group_norm_bias);

  cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

  cur = ggml_silu(ctx0, cur);

  cur = ggml_cont(
      ctx0, ggml_conv_1d(ctx0,
                         ggml_reshape_3d(ctx0, model.out_convolution_weight, 3,
                                         1024, 200),
                         cur, 1, 1, 1));

  cur = ggml_cont(
      ctx0, ggml_transpose(
                ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                               model.out_convolution_bias)));

  ggml_set_name(cur, "output");

  ggml_build_forward_expand(gf, cur);

  ggml_free(ctx0);

  return gf;
}

// clang-format off


/*
 
 ██╗   ██╗ ██████╗  ██████╗ ██████╗ ██████╗ ███████╗██████╗ 
 ██║   ██║██╔═══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗
 ██║   ██║██║   ██║██║     ██║   ██║██║  ██║█████╗  ██████╔╝
 ╚██╗ ██╔╝██║   ██║██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗
  ╚████╔╝ ╚██████╔╝╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║
   ╚═══╝   ╚═════╝  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
                                                            
  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗                  
 ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║                  
 ██║  ███╗██████╔╝███████║██████╔╝███████║                  
 ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║                  
 ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║                  
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝                  
*/

// clang-format on

struct ggml_cgraph *vocoder_graph(struct vocoder_model &model,
                                  std::vector<float> &mel_vector,
                                  std::vector<float> &padding_vector) {

  static size_t buf_size = ggml_tensor_overhead() * GPT2_MAX_NODES +
                           ggml_graph_overhead_custom(GPT2_MAX_NODES, false);
  static std::vector<uint8_t> buf(buf_size);

  struct ggml_init_params params = {
      /*.mem_size   =*/buf_size,
      /*.mem_buffer =*/buf.data(),
      /*.no_alloc   =*/true, // the tensors will be allocated later by
                             // ggml_gallocr_alloc_graph()
  };

  struct ggml_context *ctx0 = ggml_init(params);

  struct ggml_cgraph *gf = ggml_new_graph_custom(ctx0, GPT2_MAX_NODES, false);

  struct ggml_tensor *mel =
      ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, mel_vector.size() / 100, 100, 1);

  ggml_set_name(mel, "mel_tensor");

  struct ggml_tensor *padding =
      ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 10, 100, 1);

  ggml_set_name(padding, "padding_tensor");

  struct ggml_tensor *vocoder_noise = ggml_new_tensor_3d(
      ctx0, GGML_TYPE_F32, mel_vector.size() / 100 + 10, 64, 1);

  ggml_set_name(vocoder_noise, "vocoder_noise_tensor");

  ggml_build_forward_expand(gf, mel);
  ggml_build_forward_expand(gf, padding);
  ggml_build_forward_expand(gf, vocoder_noise);

  struct ggml_tensor *permuted_mel =
      ggml_cont(ctx0, ggml_permute(ctx0, mel, 2, 1, 0, 3));
  padding = ggml_cont(ctx0, ggml_permute(ctx0, padding, 2, 1, 0, 3));

  ggml_tensor *padded_mel = ggml_concat(ctx0, permuted_mel, padding, 2);

  padded_mel = ggml_cont(ctx0, ggml_permute(ctx0, padded_mel, 2, 1, 0, 3));

  ggml_tensor *cur = ggml_pad_reflect_1d(ctx0, vocoder_noise, 3, 3);

  cur = ggml_cont(
      ctx0, ggml_conv_1d(ctx0, model.convolution_pre_weight, cur, 1, 0, 1));

  cur = ggml_cont(
      ctx0, ggml_transpose(
                ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                               model.convolution_pre_bias)));

  int strides[] = {8, 8, 4};
  int paddings[] = {4, 4, 2};
  int hop_sizes[] = {8, 64, 256};
  // const int kernel_size = 3;
  // const int dilation = 1;
  const int padding_length = 1;

  struct ggml_tensor *conditioning;

  // graph tether
  // res blocks
  for (int i = 0; i < 3; i++) {

    ggml_tensor *float_32_conv_transpose_1d_weight = ggml_cont(
        ctx0,
        ggml_cpy(ctx0, model.residual_stack[i].convolution_t_pre_weight,
                 ggml_new_tensor(
                     ctx0, GGML_TYPE_F32, 4,
                     model.residual_stack[i].convolution_t_pre_weight->ne)));

    cur = ggml_cont(ctx0, ggml_leaky_relu(ctx0, cur, 0.2, false));

    cur = ggml_cont(ctx0,
                    ggml_conv_transpose_1d(
                        ctx0, model.residual_stack[i].convolution_t_pre_weight,
                        cur, strides[i], 0, 1));

    cur = ggml_cont(
        ctx0, ggml_view_2d(ctx0, cur, cur->ne[0] - (2 * paddings[i]), 32,
                           cur->nb[1], paddings[i] * ggml_element_size(cur)));

    cur = ggml_cont(
        ctx0,
        ggml_transpose(
            ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                           model.residual_stack[i].convolution_t_pre_bias)));

    conditioning =
        ggml_cpy(ctx0, padded_mel,
                 ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, padded_mel->ne));

    conditioning = ggml_cont(
        ctx0,
        ggml_conv_1d(
            ctx0,
            model.residual_stack[i].kernel_predictor_input_convolution_weight,
            conditioning, 1, 2, 1));

    conditioning = ggml_cont(
        ctx0,
        ggml_transpose(
            ctx0,
            ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, conditioning)),
                     model.residual_stack[i]
                         .kernel_predictor_input_convolution_bias)));

    conditioning =
        ggml_cont(ctx0, ggml_leaky_relu(ctx0, conditioning, 0.2, false));

    ggml_tensor *conditioning_offset;

    for (int c = 0; c < 3; c++) {

      conditioning_offset = ggml_cont(
          ctx0, ggml_conv_1d(ctx0,
                             model.residual_stack[i]
                                 .kernel_predictor_residual_conv_blocks[c]
                                 .residual_convs_1_weight,
                             conditioning, 1, 1, 1));

      conditioning_offset = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0, ggml_add(ctx0,
                             ggml_cont(ctx0, ggml_transpose(
                                                 ctx0, conditioning_offset)),
                             model.residual_stack[i]
                                 .kernel_predictor_residual_conv_blocks[c]
                                 .residual_convs_1_bias)));

      conditioning_offset = ggml_cont(
          ctx0, ggml_leaky_relu(ctx0, conditioning_offset, 0.2, false));

      conditioning_offset = ggml_cont(
          ctx0, ggml_conv_1d(ctx0,
                             model.residual_stack[i]
                                 .kernel_predictor_residual_conv_blocks[c]
                                 .residual_convs_3_weight,
                             conditioning_offset, 1, 1, 1));

      conditioning_offset = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0, ggml_add(ctx0,
                             ggml_cont(ctx0, ggml_transpose(
                                                 ctx0, conditioning_offset)),
                             model.residual_stack[i]
                                 .kernel_predictor_residual_conv_blocks[c]
                                 .residual_convs_3_bias)));

      conditioning_offset = ggml_cont(
          ctx0, ggml_leaky_relu(ctx0, conditioning_offset, 0.2, false));

      conditioning = ggml_add(ctx0, conditioning, conditioning_offset);
    }

    struct ggml_tensor *kernels = ggml_cont(
        ctx0,
        ggml_conv_1d(
            ctx0,
            model.residual_stack[i].kernel_predictor_kernel_convolution_weight,
            conditioning, 1, 1, 1));

    kernels = ggml_cont(
        ctx0,
        ggml_transpose(
            ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, kernels)),
                           model.residual_stack[i]
                               .kernel_predictor_kernel_convolution_bias)));

    struct ggml_tensor *bias = ggml_cont(
        ctx0,
        ggml_conv_1d(
            ctx0,
            model.residual_stack[i].kernel_predictor_bias_convolution_weight,
            conditioning, 1, 1, 1));

    bias = ggml_cont(
        ctx0,
        ggml_transpose(
            ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, bias)),
                           model.residual_stack[i]
                               .kernel_predictor_bias_convolution_bias)));

    kernels = ggml_reshape_3d(ctx0, kernels, kernels->ne[0], 6144, 4);
    bias = ggml_reshape_3d(ctx0, bias, bias->ne[0], 64, 4);

    int conv_block_dilations[] = {1, 3, 9, 27};
    int conv_block_paddings[] = {1, 3, 9, 27};

    ggml_tensor *output;
    ggml_tensor *k;
    ggml_tensor *b;
    ggml_tensor *reshaped_kernel;
    ggml_tensor *output_accumulator;
    ggml_tensor *output_half_1;
    ggml_tensor *output_half_2;

    for (int c = 0; c < 4; c++) {

      output = ggml_leaky_relu(ctx0, cur, 0.2, false);

      output = ggml_cont(
          ctx0,
          ggml_conv_1d(
              ctx0, model.residual_stack[i].conv_blocks[c].conv_block_1_weight,
              output, 1, conv_block_paddings[c], conv_block_dilations[c]));

      output = ggml_cont(
          ctx0,
          ggml_transpose(
              ctx0,
              ggml_add(
                  ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, output)),
                  model.residual_stack[i].conv_blocks[c].conv_block_1_bias)));

      output = ggml_cpy(ctx0, output,
                        ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, output->ne));

      output = ggml_leaky_relu(ctx0, output, 0.2, false);

      k = ggml_cont(ctx0,
                    ggml_view_3d(ctx0, kernels, kernels->ne[0], 6144, 1,
                                 kernels->nb[1], kernels->nb[2],
                                 c * kernels->ne[0] * 6144 * sizeof(float)));
      k = ggml_reshape_4d(ctx0, k, kernels->ne[0], 3, 64, 32);

      b = ggml_cont(ctx0, ggml_view_3d(ctx0, bias, bias->ne[0], 64, 1,
                                       bias->nb[1], bias->nb[2],
                                       c * bias->ne[0] * 64 * sizeof(float)));
      b = ggml_reshape_2d(ctx0, b, bias->ne[0], 64);

      output = ggml_pad_ext(ctx0, output, 1, 1, 0, 0, 0, 0, 0, 0);

      output = ggml_unfold_1d(ctx0, output, hop_sizes[i] + 2 * padding_length,
                              hop_sizes[i]);

      output = ggml_unfold_1d(ctx0, output, 1, 1);

      output = ggml_cont(ctx0, ggml_permute(ctx0, output, 1, 0, 2, 3));

      const int output_length = output->ne[2];

      output = ggml_reshape_3d(ctx0, output, hop_sizes[i] + 2 * padding_length,
                               1, output_length * 32);

      output = ggml_unfold_1d(ctx0, output, 3, 1);

      // o = torch.einsum('bildsk,biokl->bolsd', x, kernel)

      output =
          ggml_reshape_4d(ctx0, output, 3, hop_sizes[i], output_length, 32);

      reshaped_kernel = ggml_reshape_4d(ctx0, k, output_length, 3, 64, 32);

      reshaped_kernel =
          ggml_cont(ctx0, ggml_permute(ctx0, reshaped_kernel, 2, 0, 1, 3));

      output = ggml_mul_mat(ctx0, reshaped_kernel, output);

      output_accumulator = ggml_cont(
          ctx0,
          ggml_view_4d(ctx0, output, 64, hop_sizes[i], output_length, 1,
                       output->nb[1], output->nb[2], output->nb[3],
                       0 * output_length * 64 * hop_sizes[i] * sizeof(float)));
      for (int j = 1; j < 32; j++) {
        output_accumulator = ggml_add(
            ctx0, output_accumulator,
            ggml_cont(ctx0, ggml_view_4d(ctx0, output, 64, hop_sizes[i],
                                         output_length, 1, output->nb[1],
                                         output->nb[2], output->nb[3],
                                         j * output_length * 64 * hop_sizes[i] *
                                             sizeof(float))));
      }

      output =
          ggml_cont(ctx0, ggml_permute(ctx0, output_accumulator, 3, 1, 2, 0));

      output = ggml_add(ctx0, output,
                        ggml_reshape_4d(ctx0, b, 1, 1, output_length, 64));

      output =
          ggml_reshape_3d(ctx0, output, 1, hop_sizes[i] * output_length, 64);

      /*
      if (i >= 1)
      {
         cur = output;
          goto end;
      }*/

      output_half_1 = ggml_sigmoid(
          ctx0,
          ggml_cont(ctx0,
                    ggml_view_3d(ctx0, output, 1, hop_sizes[i] * output_length,
                                 32, output->nb[1], output->nb[2],
                                 0 * 1 * hop_sizes[i] * output_length * 32 *
                                     sizeof(float))));
      output_half_2 = ggml_tanh(
          ctx0,
          ggml_cont(ctx0,
                    ggml_view_3d(ctx0, output, 1, hop_sizes[i] * output_length,
                                 32, output->nb[1], output->nb[2],
                                 1 * 1 * hop_sizes[i] * output_length * 32 *
                                     sizeof(float))));

      cur = ggml_add(
          ctx0, cur,
          ggml_reshape_4d(ctx0, ggml_mul(ctx0, output_half_1, output_half_2),
                          output_length * hop_sizes[i], 32, 1, 1));
    }
  }

  cur = ggml_leaky_relu(ctx0, cur, 0.2, false);

  cur = ggml_cont(
      ctx0, ggml_conv_1d(ctx0, model.convolution_post_weight, cur, 1, 0, 1));

  cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

  cur = ggml_cont(
      ctx0, ggml_transpose(
                ctx0, ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
                               model.convolution_post_bias)));

  cur = ggml_cpy(ctx0, cur, ggml_new_tensor(ctx0, GGML_TYPE_F32, 4, cur->ne));

  ggml_build_forward_expand(gf, cur);
  ggml_set_name(cur, "vocoder_output");

  ggml_free(ctx0);

  return gf;
}

// clang-format off


/*
 
 ██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗                           
 ██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗                          
 ███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝                          
 ██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗                          
 ██║  ██║███████╗███████╗██║     ███████╗██║  ██║                          
 ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝                          
                                                                           
 ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
 ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
 █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
 ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
 ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
 ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
                                                                           
 
*/

// clang-format on

// great work openchat3.5!!!
void apply_padding(std::vector<int> &vec) {
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

void printValuesAboveThreshold(const std::vector<float> &vec, float threshold) {
  std::cout << "REACHED 6 " << std::endl;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] > threshold) {
      std::cout << "Index: " << i % 8194 << ", Value: " << vec[i] << std::endl;
    }
  }
}

std::vector<float> apply_penalty(const std::vector<float> score,
                                 float penalty) {
  std::vector<float> result(score.size());
  for (size_t i = 0; i < score.size(); ++i) {
    result[i] = (score[i] < 0) ? score[i] * penalty : score[i] / penalty;
  }
  return result;
}

std::vector<float> gather(std::vector<float> src, std::vector<int> input_ids,
                          int batch_size) {

  const int sequence_length = input_ids.size() / batch_size;
  const int vocab_size =
      src.size() / batch_size; // this is 8194, hardcoding for now

  std::vector<float> result(input_ids.size());

  for (int i = 0; i < input_ids.size(); i++) {

    const int rowIndex = i / sequence_length;

    const int colIndex = input_ids[i];

    result[i] = src[rowIndex * vocab_size + colIndex];
  }
  return result;
}

std::vector<float> scatter(std::vector<float> src1, std::vector<float> src2,
                           std::vector<int> input_ids, int batch_size) {
  std::vector<float> result;
  result.resize(src1.size());
  std::copy(src1.begin(), src1.end(), result.begin());

  const int sequence_length = input_ids.size() / batch_size;
  const int vocab_size =
      src1.size() / batch_size; // this is 8194, hardcoding for now

  // std::vector<float> result(input_ids.size());

  for (int i = 0; i < input_ids.size(); i++) {

    const int rowIndex = i / sequence_length;

    const int colIndex = input_ids[i];

    result[rowIndex * vocab_size + colIndex] = src2[i];
  }
  // printVector(result, 3, "scatter_result");
  return result;
}

void temp_inplace(std::vector<float> &src, float temp) {
  for (int i = 0; i < src.size(); i++) {
    src[i] /= temp;
  }
}

void val_where_below_thresh(std::vector<float> &src, float threshold,
                            float val) {
  for (int i = 0; i < src.size(); i++) {
    if (src[i] < threshold)
      src[i] = val;
  }
}

float nth_largest(std::vector<float> src, int n) {
  std::sort(src.begin(), src.end());
  return src[src.size() - n];
}

void top_k_inplace(std::vector<float> &src, int k) {
  k = std::min(k, 8194);
  float kth_largest_val = nth_largest(src, k);
  val_where_below_thresh(src, kth_largest_val,
                         std::numeric_limits<float>::lowest());
}

void softmax_inplace(std::vector<float> &src) {
  assert(src.size() == 8194);
  float sum = 0;
  for (int i = 0; i < src.size(); i++) {
    assert(src.size() == 8194);

    src[i] = exp(src[i]);
    sum += src[i];
  }
  for (int j = 0; j < src.size(); j++) {
    assert(src.size() == 8194);
    src[j] /= sum;
  }
}

void top_p_inplace(std::vector<float> &src) {

  std::vector<std::pair<float, int>> src_pairs;
  for (int i = 0; i < src.size(); i++) {
    src_pairs.push_back(std::make_pair(src[i], i));
  }

  std::vector<float> sorted_logits;
  std::vector<int> sorted_indices;

  std::sort(src_pairs.begin(), src_pairs.end(),
            [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
              return a.first < b.first;
            });

  for (const auto &pair : src_pairs) {
    sorted_logits.push_back(pair.first);
    sorted_indices.push_back(pair.second);
  }

  // next we perform softmax on the vector of floats sorted_logits
  assert(sorted_logits.size() == 8194);
  softmax_inplace(sorted_logits);
  // next we perform an in place cumulative sum operation on sorted logits

  for (int i = 1; i < sorted_logits.size(); i++) {
    sorted_logits[i] += sorted_logits[i - 1];
  }

  for (int i = 0; i < src.size() - 1;
       i++) // this -1 is because for some reason the last token is never zeroed
            // out.
  {
    if (sorted_logits[i] <= 0.2) {
      src[src_pairs[i].second] = std::numeric_limits<float>::lowest();
    }
  }
}

std::vector<float> sample_normal_noise(int length) {
  std::vector<float> noise(length);
  for (int i = 0; i < length; i++) {
    noise[i] = normal_distribution(generator);
  }
  return noise;
}

int multinomial(
    std::vector<float> probs) // worth changing to a binary search at some
                              // point, but for now done the simple way
{

  float sample = distribution(generator);
  sample = distribution(generator);

  float cumulative_probability = 0;
  for (int i = 0; i < probs.size(); i++) {
    cumulative_probability += probs[i];
    if (cumulative_probability >= sample) {
      return i;
    }
  }
  return 8194 -
         1; // not supposed to be reached, but defaults to the last probability
}

std::vector<int> get_relative_position_buckets(int latent_length) {

  std::vector<int> relative_positions(latent_length * latent_length);
  std::vector<int> mask(latent_length * latent_length);
  for (int i = 0; i < latent_length; i++) {
    for (int c = 0; c < latent_length; c++) {
      relative_positions[i * latent_length + c] = (abs(c - i));
      mask[i * latent_length + c] = (i < c) ? 16 : 0;

      int val_if_large =
          8 + (int)(log(float(relative_positions[i * latent_length + c]) / 8) /
                    log(64.0 / 8.0) * (16.0 - 8.0)

              );
      if (val_if_large > 15) {
        val_if_large = 15;
      }
      if (relative_positions[i * latent_length + c] < 8) {
        mask[i * latent_length + c] +=
            relative_positions[i * latent_length + c];
      } else {
        mask[i * latent_length + c] += val_if_large;
      }
    }
  }

  return mask;
}

// takes the raw logits coming out of of a pass through gpt2 and transforms them
// into a multinomial distribution, then samples from said distribution.
std::vector<int>
process_logits_and_sample(ggml_cgraph *gf,
                          std::vector<int> &mel_transformer_inputs_vector,
                          int index, int batch_size) {

  ggml_tensor *next_token_logits = gf->nodes[gf->n_nodes - 1];

  // save_f32_tensor(next_token_logits, "logs/next_token_logits_" +
  // std::to_string(index) +   ".data");

  int elements = next_token_logits->ne[0] * next_token_logits->ne[1] *
                 next_token_logits->ne[2] * next_token_logits->ne[3];

  std::vector<float> next_token_logits_vector(elements);
  ggml_backend_tensor_get(next_token_logits, next_token_logits_vector.data(), 0,
                          sizeof(float) * elements);

  std::vector<float> gather_result = gather(
      next_token_logits_vector, mel_transformer_inputs_vector, batch_size);
  gather_result = apply_penalty(gather_result, 2.0);
  std::vector<float> transformed_mel_transformer_inputs_vector =
      scatter(next_token_logits_vector, gather_result,
              mel_transformer_inputs_vector, batch_size);

  std::vector<int> samples(batch_size); // batch size is 4

  std::vector<float> probs(transformed_mel_transformer_inputs_vector.size());
  for (int i = 0; i < batch_size; i++) // hardcoded to batch size of 4
  {

    std::vector<float> logits;
    logits.insert(
        logits.end(),
        transformed_mel_transformer_inputs_vector.begin() + (i * 8194),
        transformed_mel_transformer_inputs_vector.begin() + ((i + 1) * 8194));
    assert(logits.size() == 8194);
    // transformed_mel_transformer_inputs_vector.resize(8194); // just
    // considering 1 out of 4 in the batch for now for testing purposes;
    temp_inplace(logits, 0.8);
    top_k_inplace(logits, 50);
    top_p_inplace(logits);

    softmax_inplace(
        logits); // they become probs at this point, so name is misleading now
    samples[i] = multinomial(logits);

    for (int c = 0; c < 8194; c++) {
      probs[i * 8194 + c] = logits[c];
    }
  }
  // probs
  // printValuesAboveThreshold(probs, .01);
  return samples;
}

// thanks gpt3.5
void replaceAll(std::string &str, const std::string &from,
                const std::string &to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length(); // In case 'to' contains 'from', advance start_pos
                              // to avoid infinite loop
  }
}

// thanks gpt3.5
// Function to write a WAV file from floating-point data
void writeWav(const char *filename, const std::vector<float> &data,
              int sampleRate) {
  // WAV file parameters
  int numChannels = 1;    // Mono
  int bitsPerSample = 32; // Float (32-bit)
  int byteRate = sampleRate * numChannels * bitsPerSample / 8;
  int blockAlign = numChannels * bitsPerSample / 8;

  // Open the output file
  std::ofstream outFile(filename, std::ios::binary);
  if (!outFile.is_open()) {
    std::cerr << "Error opening output file." << std::endl;
    return;
  }

  // Write the WAV header
  // RIFF header
  outFile.write("RIFF", 4);
  int fileSize =
      36 + data.size() * sizeof(float); // Size of the entire file minus 8 bytes
  outFile.write(reinterpret_cast<const char *>(&fileSize), 4);
  outFile.write("WAVE", 4);

  // Format subchunk
  outFile.write("fmt ", 4);
  int fmtSize = 16;
  outFile.write(reinterpret_cast<const char *>(&fmtSize), 4);
  int audioFormat = 3; // Floating point PCM
  outFile.write(reinterpret_cast<const char *>(&audioFormat), 2);
  outFile.write(reinterpret_cast<const char *>(&numChannels), 2);
  outFile.write(reinterpret_cast<const char *>(&sampleRate), 4);
  outFile.write(reinterpret_cast<const char *>(&byteRate), 4);
  outFile.write(reinterpret_cast<const char *>(&blockAlign), 2);
  outFile.write(reinterpret_cast<const char *>(&bitsPerSample), 2);

  // Data subchunk
  outFile.write("data", 4);
  int dataSize = data.size() * sizeof(float);
  outFile.write(reinterpret_cast<const char *>(&dataSize), 4);

  // Write the audio data
  outFile.write(reinterpret_cast<const char *>(data.data()), dataSize);

  // Close the file
  outFile.close();

  std::cout << "WAV file saved successfully. :^)" << std::endl;
}

// trims latents to no more than 8 calm tokens at the end
// the latents are each 1024 x 500, and there is a number equal to the batch
// size.
std::vector<std::vector<float>>
trim_latents(std::vector<float> &latents,
             std::vector<std::vector<int>> &mel_codes) {

  int total_sequence_length = 0;
  for (int i = 0; i < mel_codes.size(); i++) {

    // Remove first element
    mel_codes[i].erase(mel_codes[i].begin());

    // Remove last element
    mel_codes[i].erase(mel_codes[i].end() - 1);

    total_sequence_length += mel_codes[i].size();
    localAssert(mel_codes[i].size() == 500);
  }

  // localAssert (total_sequence_length == latents.size());

  std::vector<std::vector<float>> trimmed_latents(mel_codes.size());

  for (int i = 0; i < mel_codes.size(); i++) {
    const int offset = i * 500 * 1024;
    int calm_tokens = 0;

    for (int c = 0; c < 500; c++) {

      if (mel_codes[i][c] == 83) {
        calm_tokens += 1;
      } else {
        calm_tokens = 0;
      }
      if (calm_tokens > 8) {
        break;
      }
      for (int j = 0; j < 1024; j++) {
        trimmed_latents[i].push_back(latents[offset + (c * 1024) + j]);
      }
    }
    std::cout << " " << trimmed_latents[i].size() << " ";
  }
  return trimmed_latents;
}

// prints either all leaves from a computational graph, or all the nodes
void print_all_tensors(struct ggml_cgraph *gf, bool leaves, bool filter_flag,
                       std::string filter) {

  int count = leaves ? gf->n_leafs : gf->n_nodes;

  for (int i = 0; i < count; i++) {
    ggml_tensor *test;
    if (leaves) {
      test = gf->leafs[i];
    } else {
      test = gf->nodes[i];
    }

    if (filter_flag && std::string(test->name) != filter) {
      continue;
    }

    save_f32_tensor(test, "logs/" + std::string(test->name) + ".txt");

    std::cout << "---------------------------------------------------"
              << std::endl;
    std::cout << "NAME:" << std::endl;
    std::cout << test->name << std::endl;
    std::cout << "TYPE" << std::endl;
    std::cout << test->type << std::endl;
    std::cout << "SHAPE:" << std::endl;
    std::cout << test->ne[0] << std::endl;
    std::cout << test->ne[1] << std::endl;
    std::cout << test->ne[2] << std::endl;
    std::cout << test->ne[3] << std::endl;
    std::cout << "DATA:" << std::endl;

    if (!ggml_is_contiguous(test)) {
      std::cout << "Skipped data; not contiguous" << std::endl;
      continue;
    }
    if (ggml_is_transposed(test)) {
      std::cout << "Transposed:" << std::endl;
    }

    int elements = test->ne[0] * test->ne[1] * test->ne[2] * test->ne[3];

    // ggml_tensor * weights = gf->leafs[gf->n_leafs -2];
    // ggml_tensor * tokens = gf->leafs[gf->n_leafs -1];

    // ggml_graph_dump_dot(gf, NULL, "autoregressive.dot");
    // std::cout << "made it here" << std::endl;
    if (test->type == GGML_TYPE_F32) {
      std::vector<float> test_read(elements);
      ggml_backend_tensor_get(test, test_read.data(), 0,
                              sizeof(float) * elements);
      //
      for (int c = 0; c < elements; c++) {

        if (c < 3 || c > elements - 4) {

          std::cout << (test_read.data()[c]) << std::endl;
        }
      }
    } else if (test->type == GGML_TYPE_F16) {
      std::vector<ggml_fp16_t> test_read(elements);
      ggml_backend_tensor_get(test, test_read.data(), 0,
                              sizeof(ggml_fp16_t) * elements);
      //
      for (int c = 0; c < elements; c++) {
        if (c < 3 || c > elements - 4) {

          std::cout << ggml_fp16_to_fp32(test_read.data()[c]) << std::endl;
        }
      }
    } else if (test->type == GGML_TYPE_I32) {
      std::vector<int32_t> test_read(elements);
      ggml_backend_tensor_get(test, test_read.data(), 0,
                              sizeof(int32_t) * elements);
      //
      for (int c = 0; c < elements; c++) {
        if (c < 3 || c > elements - 4) {

          std::cout << test_read.data()[c] << std::endl;
        }
      }
    }
  }
}

// thanks gpt3.5!
std::vector<float> load_f32_vector(const std::string &filename, size_t nBytes) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
    return {};
  }

  // Calculate number of floats to read based on number of bytes
  size_t numFloats = nBytes / sizeof(float);
  std::vector<float> floats(numFloats);

  // Read floats from file
  file.read(reinterpret_cast<char *>(floats.data()), nBytes);

  file.close();

  return floats;
}

// thanks gpt3.5!
void progressBar(int width, int percent) {
  std::cout << "[";
  int pos = width * percent / 100;
  for (int i = 0; i < width; ++i) {
    if (i <= pos)
      std::cout << "=";
    else
      std::cout << " ";
  }
  std::cout << "] " << percent << "%\r";
  std::cout.flush();
}

void tokensSampled(int n) {
  std::cout << "tokens sampled: " << n << "\r";
  std::cout.flush();
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<int>>>
autoregressive(std::vector<gpt_vocab::id> tokens, std::string voice_path,
               int batch_size) {

  // std::string message = "this[SPACE]is[SPACE]a[SPACE]test[SPACE]message";
  // std::vector<gpt_vocab::id> tokens = ::gpt_tokenize(vocab, message);
  // std::vector<gpt_vocab::id> tokens =
  // ::parse_tokens_from_string("255,147,2,54,2,14,2,33,218,2,26,61,150,112,0,0",
  // ','); // for now, skipping some token processing steps
  // std::vector<gpt_vocab::id> tokens =
  // ::parse_tokens_from_string("255,147,2,54,2,14,2,33,218,2,26,61,150,112,0,0",
  // ','); // "This is a test message" std::vector<gpt_vocab::id> tokens =
  // ::parse_tokens_from_string("255,17,140,19,142,107,2,115,126,25,2,170,29,64,136,3,0,0",
  // ','); // "Diffusion model complete!"

  // std::vector<gpt_vocab::id> tokens =
  // ::parse_tokens_from_string("255,15,55,49,9,9,9,2,134,16,51,31,2,19,46,18,176,13,0,0",
  // ','); //"Based... Dr. Freeman?" std::vector<gpt_vocab::id> tokens =
  // ::parse_tokens_from_string("255,135,198,48,167,158,32,3,2,14,34,51,46,20,175,212,76,2,115,126,25,2,170,29,64,136,3,0,0,255,135,198,48,167,158,32,3,2,14,34,51,46,20,175,212,76,2,115,126,25,2,170,29,64,136,3,0,0",
  // ','); //"Congratulations! Autoregressive model complete!" exit(0);
  // std::vector<gpt_vocab::id> tokens =
  // ::parse_tokens_from_string("255,42,2,97,60,49,2,63,48,61,2,26,163,2,149,2,68,161,33,2,42,2,33,77,78,16,32,2,58,2,42,2,50,18,125,9,2,80,43,32,2,127,2,106,29,57,33,159,7,2,55,2,204,32,9,2,16,31,54,54,2,26,213,61,2,60,2,136,26,29,242,2,51,2,22,20,95,46,2,42,2,36,54,18,2,46,63,31,137,192,2,73,2,26,245,2,26,50,2,19,46,18,9,2,0,0",
  // ','); // "The United States must not adopt the tactics of the enemy. Means
  // are important, as ends. Crisis makes it tempting to ignore the wise
  // restraints that make men free." - Frank Church

  // exit(0);
  // todo see why this tokenization doesn't match the tokenization produced by
  // tortoise-tts (tortoise tts one does not always use the token corresponding
  // to the most characters)

  ggml_time_init();
  const int64_t t_main_start_us = ggml_time_us();

  int64_t t_load_us = 0;

  std::string file_path = "../models/ggml-model.bin";

  autoregressive_model model;

  // load the model
  {
    const int64_t t_start_us = ggml_time_us();

    if (!autoregressive_model_load(file_path, model)) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__,
              file_path.c_str());
      exit(1);
    }

    t_load_us = ggml_time_us() - t_start_us;
  }

  std::vector<int> mel_transformer_inputs_vector = std::vector<int>();
  mel_transformer_inputs_vector.resize((tokens.size() + 2) * batch_size);
  // assert(tokens.size() == 16);

  for (int i = 0; i < mel_transformer_inputs_vector.size(); i++) {
    if (i % (tokens.size() + 2) == tokens.size() + 2 - 1) {
      mel_transformer_inputs_vector[i] = 8192;
    } else {
      mel_transformer_inputs_vector[i] = 1;
    }
  }

  ggml_backend_buffer_t buf_compute;

  ggml_gallocr_t allocr = NULL;
  // allocate the compute buffer

  allocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

  struct ggml_cgraph *measure_gf = autoregressive_graph(
      model, mel_transformer_inputs_vector, tokens, true, 0, 0, batch_size);
  // ggml_graph_print(gf);

  // compute the required memory
  ggml_gallocr_reserve(allocr, measure_gf);
  size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
  // fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__,
  //         mem_size / 1024.0 / 1024.0);
  //  recreate the allocator with the required memory
  //  ggml_allocr_reset(allocr);
  //  allocr = ggml_gallocr_new(model.backend);
  //  gf = autoregressive_graph(model,mel_transformer_inputs_vector, tokens,
  //  true, 0,0); ggml_allocr_alloc_graph(allocr, gf); std::cout << "reached
  //  computing time" << std::endl;

  struct ggml_cgraph *gf = autoregressive_graph(
      model, mel_transformer_inputs_vector, tokens, true, 0, 0, batch_size);

  ggml_gallocr_alloc_graph(allocr, gf);

  struct ggml_tensor *input_tokens_tensor =
      ggml_graph_get_tensor(gf, "input_tokens");

  int token_count = tokens.size();
  ggml_backend_tensor_set(input_tokens_tensor, tokens.data(), 0,
                          token_count * ggml_element_size(input_tokens_tensor));

  struct ggml_tensor *input_position_tensor =
      ggml_graph_get_tensor(gf, "input_position");

  for (int i = 0; i < token_count; ++i) {
    int32_t v = i;
    ggml_backend_tensor_set(input_position_tensor, &v, i * sizeof(int32_t),
                            sizeof(v));
  }

  struct ggml_tensor *input_mel_tokens_tensor =
      ggml_graph_get_tensor(gf, "input_mel_tokens");

  for (int i = 0; i < mel_transformer_inputs_vector.size(); ++i) {
    int v = mel_transformer_inputs_vector[i];
    ggml_backend_tensor_set(input_mel_tokens_tensor, &v, i * sizeof(int32_t),
                            sizeof(v));
  }

  struct ggml_tensor *input_mel_tokens_truncated_tensor =
      ggml_graph_get_tensor(gf, "input_mel_tokens_truncated");

  int32_t start_mel_token = 8192;
  for (int i = 0; i < batch_size; ++i) {
    ggml_backend_tensor_set(input_mel_tokens_truncated_tensor, &start_mel_token,
                            i * sizeof(int32_t), sizeof(start_mel_token));
  }

  struct ggml_tensor *input_mel_position_tensor =
      ggml_graph_get_tensor(gf, "input_mel_position");

  int32_t v = 0;
  ggml_backend_tensor_set(input_mel_position_tensor, &v, 0, sizeof(v));

  struct ggml_tensor *auto_conditioning_tensor =
      ggml_graph_get_tensor(gf, "auto_conditioning");

  std::vector<float> auto_conditioning_vector =
      load_f32_vector(voice_path, 1024 * sizeof(float));

  ggml_backend_tensor_set(auto_conditioning_tensor,
                          auto_conditioning_vector.data(), 0,
                          1024 * ggml_element_size(auto_conditioning_tensor));

  ggml_backend_graph_compute(model.backend, gf);

  std::vector<int> samples;

  std::string sample_string;
  int stop_token = 8193;
  bool all_sequences_stopped = false;

  std::vector<std::vector<int>> sequences(batch_size);

  int i = 0;
  while (!all_sequences_stopped) {

    samples = process_logits_and_sample(gf, mel_transformer_inputs_vector, i,
                                        batch_size);

    tokensSampled(i + 1);

    sample_string = sample_string + ",[";

    int stop_token_count = 0;

    mel_transformer_inputs_vector.clear();
    for (int c = 0; c < batch_size; c++) {
      if (!(sequences[c].size() > 0 &&
            sequences[c][sequences[c].size() - 1] == stop_token)) {
        sequences[c].push_back(samples[c]);
      }
      if (samples[c] == stop_token) {
        stop_token_count += 1;
      }
      mel_transformer_inputs_vector.push_back(samples[c]);
      sample_string = sample_string + std::to_string(samples[c]) + ',';
    }
    if (stop_token_count == batch_size) {
      all_sequences_stopped = true;
    }
    sample_string = sample_string + "]";

    // ggml_allocr_reset(allocr);
    // allocr = ggml_allocr_new_from_buffer(buf_compute);
    gf = autoregressive_graph(model, mel_transformer_inputs_vector, tokens,
                              false, tokens.size() + 2 + i, i + 2, batch_size);

    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_tensor *input_mel_tokens_tensor =
        ggml_graph_get_tensor(gf, "input_mel_tokens");

    for (int i = 0; i < batch_size; ++i) {
      int v = mel_transformer_inputs_vector[i];
      ggml_backend_tensor_set(input_mel_tokens_tensor, &v, i * sizeof(int32_t),
                              sizeof(v));
    }

    ggml_tensor *input_fixed_embedding_ids_tensor =
        ggml_graph_get_tensor(gf, "input_fixed_embedding_ids");

    int v = i + 2;
    ggml_backend_tensor_set(input_fixed_embedding_ids_tensor, &v, 0, sizeof(v));

    ggml_backend_graph_compute(model.backend, gf);
    i += 1;
  }
  // ggml_allocr_free(allocr);

  // print_all_tensors(gf, false, true, "next token logits");

  for (int i = 0; i < batch_size; i++) {
    apply_padding(sequences[i]);
  }

  mel_transformer_inputs_vector.clear();
  for (int c = 0; c < batch_size; c++) {
    for (int i = 0; i < 502; i++) {
      mel_transformer_inputs_vector.push_back(sequences[c][i]);
    }
  }

  /*
  ggml_allocr_reset(allocr);
  buf_compute = ggml_backend_alloc_buffer(model.backend, mem_size);
  allocr = ggml_allocr_new_from_buffer(buf_compute);
  gf = autoregressive_graph(model, allocr,mel_transformer_inputs_vector, tokens,
  true, 0,0); ggml_allocr_alloc_graph(allocr, gf); std::cout << "reached
  computing time" << std::endl; ggml_backend_graph_compute(model.backend, gf);
  */

  ggml_gallocr_t latent_allocr = NULL;
  // allocate the compute buffer

  latent_allocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

  struct ggml_cgraph *latent_measure_gf = autoregressive_latent_graph(
      model, mel_transformer_inputs_vector, tokens, batch_size);
  // ggml_graph_print(gf);

  // compute the required memoryautoregresc__, latent_mem_size/1024.0/1024.0);

  struct ggml_cgraph *latent_gf = autoregressive_latent_graph(
      model, mel_transformer_inputs_vector, tokens, batch_size);

  token_count = tokens.size();
  int mel_token_count = mel_transformer_inputs_vector.size();

  ggml_gallocr_alloc_graph(latent_allocr, latent_gf);

  input_tokens_tensor = ggml_graph_get_tensor(latent_gf, "input_tokens");

  for (int i = 0; i < batch_size; i++) {
    ggml_backend_tensor_set(
        input_tokens_tensor, tokens.data(),
        i * token_count * ggml_element_size(input_tokens_tensor),
        token_count * ggml_element_size(input_tokens_tensor));
  }

  input_position_tensor = ggml_graph_get_tensor(latent_gf, "input_position");

  for (int i = 0; i < batch_size; i++) {
    for (int c = 0; c < token_count; c++) {
      int32_t v = c;
      ggml_backend_tensor_set(input_position_tensor, &v,
                              ((i * token_count) + c) * sizeof(int32_t),
                              sizeof(v));
    }
  }

  input_mel_tokens_tensor =
      ggml_graph_get_tensor(latent_gf, "input_mel_tokens");

  for (int i = 0; i < mel_token_count; ++i) {
    int32_t v = mel_transformer_inputs_vector[i];
    ggml_backend_tensor_set(input_mel_tokens_tensor, &v, i * sizeof(int32_t),
                            sizeof(v));
  }

  input_mel_position_tensor =
      ggml_graph_get_tensor(latent_gf, "input_mel_position");

  for (int i = 0; i < batch_size; i++) {
    for (int c = 0; c < mel_token_count / 4; c++) {
      int32_t v = c;
      ggml_backend_tensor_set(
          input_mel_position_tensor, &v,
          ((i * (mel_token_count / 4)) + c) * sizeof(int32_t), sizeof(v));
    }
  }

  auto_conditioning_tensor =
      ggml_graph_get_tensor(latent_gf, "auto_conditioning");

  ggml_backend_tensor_set(auto_conditioning_tensor,
                          auto_conditioning_vector.data(), 0,
                          1024 * ggml_element_size(auto_conditioning_tensor));

  ggml_backend_graph_compute(model.backend, latent_gf);

  // print_all_tensors(latent_gf, true, true, "cur");
  // print_all_tensors(latent_gf, false, true, "cur");

  // compare_to_saved_tensor_with_name(ggml_graph_get_tensor(latent_gf, "cur"));
  // exit(0);

  std::vector<float> latents = std::vector<float>();

  extract_tensor_to_vector(latent_gf->nodes[latent_gf->n_nodes - 1], latents);

  ggml_gallocr_free(latent_allocr);
  ggml_gallocr_free(allocr);

  ggml_free(model.ctx);

  ggml_backend_buffer_free(model.buffer_w);

  ggml_backend_free(model.backend);

  std::vector<std::vector<float>> trimmed_latents =
      trim_latents(latents, sequences);

  return std::make_pair(trimmed_latents, sequences);
}

// thanks gpt3.5!
std::vector<double>
get_alphas_cumulative_product(const std::vector<double> &betas) {
  std::vector<double> alphas;
  alphas.reserve(betas.size());

  // Calculate alphas
  for (const auto &beta : betas) {
    alphas.push_back(1.0f - beta);
  }

  // Calculate cumulative product
  std::vector<double> alpha_cumulative_products(alphas.size());
  std::partial_sum(alphas.begin(), alphas.end(),
                   alpha_cumulative_products.begin(),
                   std::multiplies<double>());

  return alpha_cumulative_products;
}

// thanks gpt3.5!
void get_beta_schedule(int num_diffusion_timesteps,
                       std::vector<double> &betas) {
  double scale = 1000.0 / num_diffusion_timesteps;
  double beta_start = scale * 0.0001;
  double beta_end = scale * 0.02;

  for (int i = 0; i < num_diffusion_timesteps; ++i) {
    betas.push_back(beta_start + i * (float)(beta_end - beta_start) /
                                     (num_diffusion_timesteps - 1));
  }
}

// thanks gpt3.5!
std::vector<double> sqrt(const std::vector<double> &vec) {
  std::vector<double> result(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    result[i] = std::sqrt(vec[i]);
  }
  return result;
}

// thanks gpt3.5!
std::vector<double> log(const std::vector<double> &vec) {
  std::vector<double> result(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    result[i] = std::log(vec[i]);
  }
  return result;
}

// thanks gpt3.5!
std::vector<double> ones_minus(const std::vector<double> &vec) {
  std::vector<double> result(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    result[i] = 1.0f - vec[i];
  }
  return result;
}

// thanks gpt3.5!
std::vector<double> reciprocal(const std::vector<double> &vec) {
  std::vector<double> result(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    result[i] = 1.0f / vec[i];
  }
  return result;
}

// thanks gpt3.5!
std::vector<double> reciprocal_minus_one(const std::vector<double> &vec) {
  std::vector<double> result(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    result[i] = 1.0f / vec[i] - 1;
  }
  return result;
}

// thanks gpt3.5!
void calculate_posterior_variance(
    std::vector<double> &posterior_variance, const std::vector<double> &betas,
    const std::vector<double> &alphas_cumprod_prev,
    const std::vector<double> &alphas_cumprod) {
  posterior_variance.resize(betas.size());
  for (size_t i = 0; i < betas.size(); ++i) {
    posterior_variance[i] =
        betas[i] * (1.0 - alphas_cumprod_prev[i]) / (1.0 - alphas_cumprod[i]);
  }
}

// thanks gpt3.5!
void calculate_posterior_log_variance_clipped(
    std::vector<double> &posterior_log_variance_clipped,
    const std::vector<double> &posterior_variance) {
  posterior_log_variance_clipped.resize(posterior_variance.size());
  posterior_log_variance_clipped[0] = std::log(posterior_variance[1]);
  for (size_t i = 1; i < posterior_variance.size(); ++i) {
    posterior_log_variance_clipped[i] = std::log(posterior_variance[i]);
  }
}

// thanks gpt3.5!
void calculate_posterior_mean_coef1(
    std::vector<double> &posterior_mean_coef1, const std::vector<double> &betas,
    const std::vector<double> &alphas_cumprod_prev,
    const std::vector<double> &alphas_cumprod) {
  posterior_mean_coef1.resize(betas.size());
  for (size_t i = 0; i < betas.size(); ++i) {
    posterior_mean_coef1[i] =
        betas[i] * sqrt(alphas_cumprod_prev[i]) / (1.0 - alphas_cumprod[i]);
  }
}

// thanks gpt3.5!
void calculate_posterior_mean_coef2(
    std::vector<double> &posterior_mean_coef2,
    const std::vector<double> &alphas_cumprod_prev,
    const std::vector<double> &alphas_cumprod,
    const std::vector<double> &betas) {
  posterior_mean_coef2.resize(alphas_cumprod_prev.size());
  for (size_t i = 0; i < alphas_cumprod_prev.size(); ++i) {
    posterior_mean_coef2[i] = (1.0 - alphas_cumprod_prev[i]) *
                              sqrt(1.0 - betas[i]) / (1.0 - alphas_cumprod[i]);
  }
}

// thanks gpt3.5!
std::vector<float>
generate_timestep_embedding(const std::vector<int> &timesteps, const int dim,
                            const int max_period) {
  int half = dim / 2;
  std::vector<float> cos_embedding;
  std::vector<float> sin_embedding;

  // Calculate frequencies
  for (int i = 0; i < half; ++i) {
    float freq = exp(-log(max_period) * static_cast<float>(i) / half);
    float arg = static_cast<float>(timesteps[0]) * freq;

    // Append cosine and sine values
    cos_embedding.push_back(cos(arg));
    sin_embedding.push_back(sin(arg));
  }

  cos_embedding.insert(cos_embedding.end(), sin_embedding.begin(),
                       sin_embedding.end());
  // If dim is odd, add one zero value to the embedding
  if (dim % 2) {
    cos_embedding.push_back(0.0f);
  }

  return cos_embedding;
}

// thanks openchat3.6
void calculate_model_variance(std::vector<float> &model_var_values,
                              std::vector<float> &log_variance, float max_log,
                              float min_log) {
  log_variance.clear();
  for (size_t i = 0; i < model_var_values.size(); ++i) {
    float frac = (model_var_values[i] + 1) / 2;
    float model_log_variance = frac * max_log + (1 - frac) * min_log;
    log_variance.push_back(model_log_variance);
    model_var_values[i] = std::exp(model_log_variance);
  }
}

// thanks openchat3.6
void blend_output_with_unconditioned_output(
    std::vector<float> &model_output,
    const std::vector<float> &model_output_no_conditioning, float cfk) {
  for (size_t i = 0; i < model_output.size(); ++i) {
    model_output[i] =
        (1 + cfk) * model_output[i] - cfk * model_output_no_conditioning[i];
  }
}

// thanks openchat3.6
std::vector<float> predict_xstart_from_eps(float sqrt_recip_alphas_cumprod,
                                           float sqrt_recipm1_alphas_cumprod,
                                           const std::vector<float> &x,
                                           const std::vector<float> &eps) {
  if (x.size() != eps.size()) {
    throw std::invalid_argument("The size of the eps vector must be the same "
                                "as the size of the x vector.");
  }

  std::vector<float> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] =
        sqrt_recip_alphas_cumprod * x[i] - sqrt_recipm1_alphas_cumprod * eps[i];
  }
  // printVector(result, 3, "prior to clamp");
  for (size_t i = 0; i < x.size(); ++i) {
    if (result[i] > 1.0) {
      result[i] = 1.0;
    }
    if (result[i] < -1.0) {
      result[i] = -1.0;
    }
  }

  return result;
}

// thanks openchat3.6!
void denormalize_tacotron_mel(std::vector<float> &norm_mel) {

  const float TACOTRON_MEL_MAX = 2.3143386840820312;
  const float TACOTRON_MEL_MIN = -11.512925148010254;

  for (auto &mel : norm_mel) {
    mel = ((mel + 1) / 2) * (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN) +
          TACOTRON_MEL_MIN;
  }
}

std::vector<float> q_posterior_mean(float posterior_mean_coef1,
                                    float posterior_mean_coef2,
                                    const std::vector<float> &x,
                                    const std::vector<float> &x_start) {
  if (x.size() != x_start.size()) {
    throw std::invalid_argument("The size of the x_start vector must be the "
                                "same as the size of the x vector.");
  }

  std::vector<float> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = posterior_mean_coef1 * x_start[i] + posterior_mean_coef2 * x[i];
  }

  return result;
}

// thanks openchat3.6
std::vector<float> sample_function(const std::vector<float> &mean,
                                   const std::vector<float> &log_variance,
                                   std::vector<float> &noise) {
  std::vector<float> result;
  for (size_t i = 0; i < mean.size(); ++i) {
    result.push_back(mean[i] + std::exp(0.5 * log_variance[i]) * noise[i]);
  }
  return result;
}

std::vector<float> diffusion(std::vector<float> trimmed_latents) {

  int length = trimmed_latents.size() / 1024;
  int output_sequence_length = length * 4 * 24000 / 22050;

  std::vector<int> output_shape;

  output_shape.push_back(1);
  output_shape.push_back(100);
  output_shape.push_back(output_sequence_length);

  std::string diffusion_file_path = "../models/ggml-diffusion-model.bin";

  diffusion_model dfsn_model;

  // load the model
  {
    if (!diffusion_model_load(diffusion_file_path, dfsn_model)) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__,
              diffusion_file_path.c_str());
      exit(1);
    }
  }

  std::vector<float> noise = sample_normal_noise(100 * output_sequence_length);
  // save_f32_vector("./logs/diffusion_noise.bin", noise);

  std::vector<int> timestep_map = {
      0,    51,   101,  152,  202,  253,  304,  354,  405,  456,  506,  557,
      607,  658,  709,  759,  810,  861,  911,  962,  1012, 1063, 1114, 1164,
      1215, 1266, 1316, 1367, 1417, 1468, 1519, 1569, 1620, 1670, 1721, 1772,
      1822, 1873, 1924, 1974, 2025, 2075, 2126, 2177, 2227, 2278, 2329, 2379,
      2430, 2480, 2531, 2582, 2632, 2683, 2733, 2784, 2835, 2885, 2936, 2987,
      3037, 3088, 3138, 3189, 3240, 3290, 3341, 3392, 3442, 3493, 3543, 3594,
      3645, 3695, 3746, 3797, 3847, 3898, 3948, 3999};

  std::vector<double> beta_schedule(0);

  int diffusion_timesteps = 80;

  float base_conditioning_free_k = 2.0;

  get_beta_schedule(4000, beta_schedule);

  // printVector(beta_schedule, 3,  "beta schedule");

  std::vector<double> alpha_cumulative_products =
      get_alphas_cumulative_product(beta_schedule);

  float last_alpha_cumulative_product = 1.0;
  beta_schedule.clear();

  for (int i : timestep_map) {
    beta_schedule.push_back(
        1 - (alpha_cumulative_products[i] / last_alpha_cumulative_product));
    last_alpha_cumulative_product = alpha_cumulative_products[i];
  }

  alpha_cumulative_products = get_alphas_cumulative_product(beta_schedule);

  // printVector(alpha_cumulative_products, 3,  "alpha cumulative products");

  std::vector<double> alpha_cumulative_products_prev(
      alpha_cumulative_products.size());
  std::vector<double> alpha_cumulative_products_next(
      alpha_cumulative_products.size());

  alpha_cumulative_products_prev.front() = 1.0f;
  std::copy(alpha_cumulative_products.begin(),
            alpha_cumulative_products.end() - 1,
            alpha_cumulative_products_prev.begin() + 1);

  std::copy(alpha_cumulative_products.begin() + 1,
            alpha_cumulative_products.end(),
            alpha_cumulative_products_next.begin());
  alpha_cumulative_products_next.back() = 0.0f;

  std::vector<double> sqrt_alphas_cumprod = sqrt(alpha_cumulative_products);
  std::vector<double> sqrt_one_minus_alphas_cumprod =
      sqrt(ones_minus(alpha_cumulative_products));
  std::vector<double> log_one_minus_alphas_cumprod =
      log(ones_minus(alpha_cumulative_products));
  std::vector<double> sqrt_reciprocal_alphas_cumprod =
      sqrt(reciprocal(alpha_cumulative_products));
  std::vector<double> sqrt_reciprocal_minus_one_alphas_cumprod =
      sqrt(reciprocal_minus_one(alpha_cumulative_products));

  std::vector<double> posterior_variance;
  std::vector<double> posterior_log_variance_clipped;
  std::vector<double> posterior_mean_coef1;
  std::vector<double> posterior_mean_coef2;

  calculate_posterior_variance(posterior_variance, beta_schedule,
                               alpha_cumulative_products_prev,
                               alpha_cumulative_products);
  calculate_posterior_log_variance_clipped(posterior_log_variance_clipped,
                                           posterior_variance);
  calculate_posterior_mean_coef1(posterior_mean_coef1, beta_schedule,
                                 alpha_cumulative_products_prev,
                                 alpha_cumulative_products);
  calculate_posterior_mean_coef2(posterior_mean_coef2,
                                 alpha_cumulative_products_prev,
                                 alpha_cumulative_products, beta_schedule);

  // ggml_backend_t temp_backend = ggml_backend_cuda_init();

  std::vector<float> x;
  x = noise;

  for (int diffusion_index = 0; diffusion_index < 80; diffusion_index++)

  {

    ggml_gallocr_t diffusion_allocr = NULL;

    std::vector<float> model_output;
    std::vector<float> model_output_no_conditioning;

    // diffusion with conditioning
    {
      diffusion_allocr = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(dfsn_model.backend));

      // alignment required by the backend
      //  size_t diffusion_align =
      //  ggml_backend_get_alignment(dfsn_model.backend); std::cout <<
      //  "alignment" << std::endl;
      // std::cout << diffusion_align << std::endl;
      // diffusion_allocr = ggml_allocr_new_measure(diffusion_align);
      // std::cout << "align created" << std::endl;

      // create the worst case graph for memory usage estimation
      // int n_tokens = std::min(model.hparams.n_ctx, params.n_batch);
      // int n_past = model.hparams.n_ctx - n_tokens;
      // ggml_allocr_reset(diffusion_allocr);
      struct ggml_cgraph *measure_gf =
          diffusion_graph(dfsn_model, trimmed_latents, output_sequence_length,
                          diffusion_index, false);
      // ggml_graph_print(gf);

      // compute the required memory
      size_t diffusion_mem_size =
          ggml_gallocr_get_buffer_size(diffusion_allocr, 0);

      ggml_gallocr_reserve(diffusion_allocr, measure_gf);
      size_t mem_size = ggml_gallocr_get_buffer_size(diffusion_allocr, 0);
      // fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__,
      //         mem_size / 1024.0 / 1024.0);

      struct ggml_cgraph *diffusion_gf =
          diffusion_graph(dfsn_model, trimmed_latents, output_sequence_length,
                          diffusion_index, false);
      ggml_gallocr_alloc_graph(diffusion_allocr, diffusion_gf);

      struct ggml_tensor *latent_tensor =
          ggml_graph_get_tensor(diffusion_gf, "input_latent_tensor");

      ggml_backend_tensor_set(latent_tensor, trimmed_latents.data(), 0,
                              trimmed_latents.size() *
                                  ggml_element_size(latent_tensor));

      struct ggml_tensor *conditioning_scale_offset_tensor =
          ggml_graph_get_tensor(diffusion_gf, "conditioning_scale_offset");

      float offset_val = 1.0;
      ggml_backend_tensor_set(conditioning_scale_offset_tensor, &offset_val, 0,
                              ggml_element_size(latent_tensor));

      std::vector<int> relative_position_buckets =
          get_relative_position_buckets(trimmed_latents.size() / 1024);

      struct ggml_tensor *relative_position_buckets_tensor =
          ggml_graph_get_tensor(diffusion_gf,
                                "relative_position_buckets_tensor");

      ggml_backend_tensor_set(
          relative_position_buckets_tensor, relative_position_buckets.data(), 0,
          (trimmed_latents.size() / 1024) * (trimmed_latents.size() / 1024) *
              ggml_element_size(relative_position_buckets_tensor));

      std::vector<int> output_relative_position_buckets =
          get_relative_position_buckets(output_sequence_length);

      struct ggml_tensor *output_relative_position_buckets_tensor =
          ggml_graph_get_tensor(diffusion_gf,
                                "output_relative_position_buckets_tensor");

      ggml_backend_tensor_set(
          output_relative_position_buckets_tensor,
          output_relative_position_buckets.data(), 0,
          (output_sequence_length * output_sequence_length) *
              ggml_element_size(output_relative_position_buckets_tensor));

      ggml_tensor *noise_tensor =
          ggml_graph_get_tensor(diffusion_gf, "noise_tensor");
      ggml_backend_tensor_set(noise_tensor, x.data(), 0,
                              (output_sequence_length * 100) *
                                  ggml_element_size(noise_tensor));

      // Vector to store vectors of floats
      std::vector<std::vector<float>> time_embeddings;
      int max_period = 10000;
      int dim = 1024;

      // Iterate backwards over the timestep_map
      for (auto it = timestep_map.rbegin(); it != timestep_map.rend(); ++it) {
        std::vector<int> timesteps = {
            *it}; // Create a vector with single timestep
        std::vector<float> embedding =
            generate_timestep_embedding(timesteps, dim, max_period);
        time_embeddings.push_back(embedding);
      }

      for (int i = 0; i < time_embeddings.size(); i++) {
        struct ggml_tensor *time_embedding_tensor = ggml_graph_get_tensor(
            diffusion_gf, ("time_embedding_" + std::to_string(i)).c_str());

        ggml_backend_tensor_set(time_embedding_tensor,
                                time_embeddings[i].data(), 0,
                                (time_embeddings[i].size() *
                                 ggml_element_size(time_embedding_tensor)));
      }

      // ggml_gallocr_alloc_graph(diffusion_allocr, diffusion_gf);
      ggml_backend_graph_compute(dfsn_model.backend, diffusion_gf);

      extract_tensor_to_vector(ggml_graph_get_tensor(diffusion_gf, "output"),
                               model_output);

      // print_all_tensors(diffusion_gf, false, true, "output");
      // print_all_tensors(diffusion_gf, true, true, "output");

      ggml_gallocr_free(diffusion_allocr);
    }

    // diffusion without conditioning
    {
      diffusion_allocr = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(dfsn_model.backend));

      // alignment required by the backend
      //  size_t diffusion_align =
      //  ggml_backend_get_alignment(dfsn_model.backend); std::cout <<
      //  "alignment" << std::endl;
      // std::cout << diffusion_align << std::endl;
      // diffusion_allocr = ggml_allocr_new_measure(diffusion_align);
      // std::cout << "align created" << std::endl;

      // create the worst case graph for memory usage estimation
      // int n_tokens = std::min(model.hparams.n_ctx, params.n_batch);
      // int n_past = model.hparams.n_ctx - n_tokens;
      // ggml_allocr_reset(diffusion_allocr);
      struct ggml_cgraph *measure_gf =
          diffusion_graph(dfsn_model, trimmed_latents, output_sequence_length,
                          diffusion_index, true);
      // ggml_graph_print(gf);

      // compute the required memory
      size_t diffusion_mem_size =
          ggml_gallocr_get_buffer_size(diffusion_allocr, 0);

      ggml_gallocr_reserve(diffusion_allocr, measure_gf);
      size_t mem_size = ggml_gallocr_get_buffer_size(diffusion_allocr, 0);
      // fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__,
      //        mem_size / 1024.0 / 1024.0);

      struct ggml_cgraph *diffusion_gf =
          diffusion_graph(dfsn_model, trimmed_latents, output_sequence_length,
                          diffusion_index, true);
      ggml_gallocr_alloc_graph(diffusion_allocr, diffusion_gf);

      struct ggml_tensor *latent_tensor =
          ggml_graph_get_tensor(diffusion_gf, "input_latent_tensor");

      ggml_backend_tensor_set(latent_tensor, trimmed_latents.data(), 0,
                              trimmed_latents.size() *
                                  ggml_element_size(latent_tensor));

      struct ggml_tensor *conditioning_scale_offset_tensor =
          ggml_graph_get_tensor(diffusion_gf, "conditioning_scale_offset");

      float offset_val = 1.0;
      ggml_backend_tensor_set(conditioning_scale_offset_tensor, &offset_val, 0,
                              ggml_element_size(latent_tensor));

      std::vector<int> relative_position_buckets =
          get_relative_position_buckets(trimmed_latents.size() / 1024);

      struct ggml_tensor *relative_position_buckets_tensor =
          ggml_graph_get_tensor(diffusion_gf,
                                "relative_position_buckets_tensor");

      ggml_backend_tensor_set(
          relative_position_buckets_tensor, relative_position_buckets.data(), 0,
          (trimmed_latents.size() / 1024) * (trimmed_latents.size() / 1024) *
              ggml_element_size(relative_position_buckets_tensor));

      std::vector<int> output_relative_position_buckets =
          get_relative_position_buckets(output_sequence_length);

      struct ggml_tensor *output_relative_position_buckets_tensor =
          ggml_graph_get_tensor(diffusion_gf,
                                "output_relative_position_buckets_tensor");

      ggml_backend_tensor_set(
          output_relative_position_buckets_tensor,
          output_relative_position_buckets.data(), 0,
          (output_sequence_length * output_sequence_length) *
              ggml_element_size(output_relative_position_buckets_tensor));

      ggml_tensor *noise_tensor =
          ggml_graph_get_tensor(diffusion_gf, "noise_tensor");
      ggml_backend_tensor_set(noise_tensor, x.data(), 0,
                              (output_sequence_length * 100) *
                                  ggml_element_size(noise_tensor));

      // Vector to store vectors of floats
      std::vector<std::vector<float>> time_embeddings;
      int max_period = 10000;
      int dim = 1024;

      // Iterate backwards over the timestep_map
      for (auto it = timestep_map.rbegin(); it != timestep_map.rend(); ++it) {
        std::vector<int> timesteps = {
            *it}; // Create a vector with single timestep
        std::vector<float> embedding =
            generate_timestep_embedding(timesteps, dim, max_period);
        time_embeddings.push_back(embedding);
      }

      for (int i = 0; i < time_embeddings.size(); i++) {
        struct ggml_tensor *time_embedding_tensor = ggml_graph_get_tensor(
            diffusion_gf, ("time_embedding_" + std::to_string(i)).c_str());

        ggml_backend_tensor_set(time_embedding_tensor,
                                time_embeddings[i].data(), 0,
                                (time_embeddings[i].size() *
                                 ggml_element_size(time_embedding_tensor)));
      }

      // ggml_gallocr_alloc_graph(diffusion_allocr, diffusion_gf);
      ggml_backend_graph_compute(dfsn_model.backend, diffusion_gf);

      // print_all_tensors(diffusion_gf, false, true, "output");
      // print_all_tensors(diffusion_gf, true, true, "output");

      extract_tensor_to_vector(ggml_graph_get_tensor(diffusion_gf, "output"),
                               model_output_no_conditioning);

      ggml_gallocr_free(diffusion_allocr);
    }

    // printVector(model_output, 3, "model_output");
    // printVector(model_output_no_conditioning, 3,
    //            "model_output_no_conditioning");

    int split_index = 100 * output_sequence_length;

    std::vector<float> model_output_means(model_output.begin(),
                                          model_output.begin() + split_index);
    std::vector<float> model_output_vars(model_output.begin() + split_index,
                                         model_output.end());
    std::vector<float> model_output_no_conditioning_means(
        model_output_no_conditioning.begin(),
        model_output_no_conditioning.begin() + split_index);

    // printVector(model_output_means, 3, "model_output_means");
    // printVector(model_output_no_conditioning_means, 3,
    //             "model_output_no_conditioning_means");
    // printVector(model_output_vars, 3, "model_output_vars");

    // printVector(posterior_log_variance_clipped, 3,
    //             "Posterior Log Variance Clipped");

    float max_log = log(beta_schedule[79 - diffusion_index]);

    float min_log = posterior_log_variance_clipped[79 - diffusion_index];

    float conditioning_free_k =
        base_conditioning_free_k *
        (1 - (float)(79 - diffusion_index) / float(diffusion_timesteps));

    std::vector<float> model_log_variance;

    calculate_model_variance(model_output_vars, model_log_variance, min_log,
                             max_log);
    // printVector(model_output_vars, 3, "model_output_vars");

    blend_output_with_unconditioned_output(model_output_means,
                                           model_output_no_conditioning_means,
                                           conditioning_free_k);
    // printVector(model_output_means, 3, "model_output_means");

    std::vector<float> x_start_pred = predict_xstart_from_eps(
        sqrt_reciprocal_alphas_cumprod[79 - diffusion_index],
        sqrt_reciprocal_minus_one_alphas_cumprod[79 - diffusion_index], x,
        model_output_means);
    // printVector(x_start_pred, 3, "x_start_pred");

    std::vector<float> final_model_mean = q_posterior_mean(
        posterior_mean_coef1[79 - diffusion_index],
        posterior_mean_coef2[79 - diffusion_index], x, x_start_pred);
    // printVector(final_model_mean, 3, "final_model_mean");

    std::vector<float> model_sample;

    std::vector<float> sample_noise =
        sample_normal_noise(100 * output_sequence_length);

    if (79 - diffusion_index != 0) {
      model_sample =
          sample_function(final_model_mean, model_log_variance, sample_noise);
    } else {
      model_sample = final_model_mean;
    }
    // printVector(model_sample, 3, "model_sample");
    x = model_sample;

    progressBar(50, (int)(((float)diffusion_index / 80.0) * 100));
  }

  ggml_free(dfsn_model.ctx);

  ggml_backend_buffer_free(dfsn_model.buffer_w);

  ggml_backend_free(dfsn_model.backend);

  return x;
}

std::vector<float> vocoder(std::vector<float> mel) {

  std::string vocoder_model_file_path = "../models/ggml-vocoder-model.bin";

  denormalize_tacotron_mel(mel);
  // printVector(mel, 3, "denormalized_mel");

  std::vector<float> padding(1000);
  for (int i = 0; i < 1000; i++) {
    padding[i] = -11.5129;
  }

  // while(true);

  std::vector<float> vocoder_noise =
      sample_normal_noise(((mel.size() / 100) + 10) * 64);
  // save_f32_vector("./logs/vocoder-noise.bin", vocoder_noise);

  vocoder_model vocoder;

  // load the model
  {
    if (!vocoder_model_load(vocoder_model_file_path, vocoder)) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__,
              vocoder_model_file_path.c_str());
      exit(1);
    }
  }

  ggml_gallocr_t vocoder_allocr = NULL;

  vocoder_allocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(vocoder.backend));

  struct ggml_cgraph *measure_gf = vocoder_graph(vocoder, mel, padding);

  // std::cout << "graph created" << std::endl;
  //  compute the required memory
  size_t vocoder_mem_size = ggml_gallocr_get_buffer_size(vocoder_allocr, 0);

  ggml_gallocr_reserve(vocoder_allocr, measure_gf);
  size_t mem_size = ggml_gallocr_get_buffer_size(vocoder_allocr, 0);
  fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__,
          mem_size / 1024.0 / 1024.0);

  struct ggml_cgraph *vocoder_gf = vocoder_graph(vocoder, mel, padding);
  ggml_gallocr_alloc_graph(vocoder_allocr, vocoder_gf);

  struct ggml_tensor *mel_tensor =
      ggml_graph_get_tensor(vocoder_gf, "mel_tensor");

  ggml_backend_tensor_set(mel_tensor, mel.data(), 0,
                          mel.size() * ggml_element_size(mel_tensor));

  struct ggml_tensor *padding_tensor =
      ggml_graph_get_tensor(vocoder_gf, "padding_tensor");

  ggml_backend_tensor_set(padding_tensor, padding.data(), 0,
                          1000 * ggml_element_size(padding_tensor));

  struct ggml_tensor *noise_tensor =
      ggml_graph_get_tensor(vocoder_gf, "vocoder_noise_tensor");

  ggml_backend_tensor_set(noise_tensor, vocoder_noise.data(), 0,
                          vocoder_noise.size() *
                              ggml_element_size(padding_tensor));

  // std::cout << "reached computing time" << std::endl;
  ggml_backend_graph_compute(vocoder.backend, vocoder_gf);

  // extract_tensor_to_vector(ggml_graph_get_tensor(vocoder_gf, "output"),
  // model_output);
  // std::cout << "reached" << std::endl;
  // print_all_tensors(vocoder_gf, false, true, "vocoder_output");
  // print_all_tensors(vocoder_gf, true, true, "vocoder_output");

  std::vector<float> audio = std::vector<float>();

  extract_tensor_to_vector(vocoder_gf->nodes[vocoder_gf->n_nodes - 1], audio);

  ggml_gallocr_free(vocoder_allocr);

  return audio;
}

// clang-format off
/*

 ████████╗███████╗███████╗████████╗██╗███╗   ██╗ ██████╗
 ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██║████╗  ██║██╔════╝
    ██║   █████╗  ███████╗   ██║   ██║██╔██╗ ██║██║  ███╗
    ██║   ██╔══╝  ╚════██║   ██║   ██║██║╚██╗██║██║   ██║
    ██║   ███████╗███████║   ██║   ██║██║ ╚████║╚██████╔╝
    ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝


*/
// clang-format on

// thanks gpt3.5 !
void save_f32_vectors(const std::string &filename,
                      const std::vector<std::vector<float>> &vectors) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << " for writing."
              << std::endl;
    return;
  }

  // Write each vector
  for (const auto &vec : vectors) {
    size_t numFloats = vec.size();
    // Write vector elements
    file.write(reinterpret_cast<const char *>(vec.data()),
               numFloats * sizeof(float));
  }

  file.close();
}

void save_f32_vector(const std::string &filename,
                     const std::vector<float> &vector) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << " for writing."
              << std::endl;
    return;
  }

  // Write each vector
  size_t numFloats = vector.size();
  // Write vector elements
  file.write(reinterpret_cast<const char *>(vector.data()),
             numFloats * sizeof(float));

  file.close();
}

// thanks gpt3.5 !
bool latent_vectors_match(const std::vector<std::vector<float>> &vec_of_vecs,
                          const std::vector<float> &other_vec) {
  // Flatten the vector of vectors
  std::vector<float> flattened;
  for (const auto &inner_vec : vec_of_vecs) {
    flattened.insert(flattened.end(), inner_vec.begin(), inner_vec.end());
  }

  // Check if lengths match
  if (flattened.size() != other_vec.size()) {
    std::cout << "size problem" << std::endl;
    return false;
  }

  // Check if each entry matches
  for (int i = 0; i < flattened.size(); i++) {
    // std::cout << std::to_string(i) + ": "  << flattened[i]  << " " <<
    // other_vec[i] << std::endl;
    if (abs(flattened[i] - other_vec[i]) > .01) {
      std::cout << flattened[i] << ":" << other_vec[i] << std::endl;
      return false;
    }
  }

  return true;
}

// thanks gpt 3.6
bool vectors_match(const std::vector<float> &vec1,
                   const std::vector<float> &vec2) {
  // Check if lengths match
  if (vec1.size() != vec2.size()) {
    std::cout << "size problem: " << vec1.size() << " " << vec2.size()
              << std::endl;

    return false;
  }

  // Check if each entry matches
  for (int i = 0; i < vec1.size(); i++) {
    if (abs(vec1[i] - vec2[i]) > .01) {
      std::cout << "i: " << i << std::endl;
      std::cout << vec1[i] << ":" << vec2[i] << std::endl;
      return false;
    }
  }

  return true;
}

bool mel_code_vectors_match(const std::vector<std::vector<int>> &vec1,
                            const std::vector<std::vector<int>> &vec2) {
  if (vec1.size() != vec2.size()) {
    return false;
  }

  for (size_t i = 0; i < vec1.size(); ++i) {
    if (vec1[i].size() != vec2[i].size()) {
      std::cout << "size: " << vec1[i].size() << " " << vec2[i].size()
                << std::endl;
      return false;
    }

    for (size_t j = 0; j < vec1[i].size(); ++j) {
      if (vec1[i][j] != vec2[i][j]) {
        return false;
      }
    }
  }

  return true;
}

void test_autoregressive() {

  // loading rng state
  //  https://stackoverflow.com/a/42386852 used for reference
  std::ifstream fin("../assets/test_autoregressive_seed.bin");
  fin >> generator;
  fin.close();
  std::ifstream fin2("../assets/test_autoregressive_distribution.bin");
  fin2 >> distribution;
  fin2.close();

  std::vector<gpt_vocab::id> tokens = ::parse_tokens_from_string(
      "255,15,55,49,9,9,9,2,134,16,51,31,2,19,46,18,176,13,0,0",
      ','); //"Based... Dr. Freeman?"

  std::pair<std::vector<std::vector<float>>, std::vector<std::vector<int>>>
      autoregressive_result = autoregressive(tokens, "../models/mol.bin", 4);

  std::vector<std::vector<float>> trimmed_latents = autoregressive_result.first;
  std::vector<std::vector<int>> sequences = autoregressive_result.second;

  int trimmed_latents_size = 0;
  for (std::vector<float> trimmed_latent : trimmed_latents) {
    trimmed_latents_size += trimmed_latent.size();
  }

  // save_f32_vectors("../assets/target_trimmed_latents.bin", trimmed_latents);
  std::vector<float> target_trimmed_latents = load_f32_vector(
      "../assets/target_trimmed_latents.bin",
      trimmed_latents_size *
          sizeof(float)); // 4 is the number of bytes in a float.

  std::vector<std::vector<int>> target_sequences = {
      {8,   7406, 6450, 1601, 2061, 4389, 4954, 134,  1554, 372,  3666, 1580,
       20,  83,   45,   8,    248,  8012, 2483, 7396, 37,   7784, 3008, 1126,
       283, 1609, 2376, 2061, 4992, 3330, 1350, 469,  1022, 7005, 8193, 83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,  83,   83,   83,   83,   45,   45,   248},
      {45,   7005, 5594, 944, 4825, 3487, 4389, 1272, 456,  2068, 4685, 1981,
       1656, 1580, 20,   45,  7406, 3386, 3932, 2483, 7683, 6893, 7136, 3221,
       3069, 734,  511,  485, 1105, 1805, 4040, 2613, 386,  497,  152,  8193,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,  83,   45,   45,   248},
      {20,   299,  7184, 2968, 3633, 3487, 7358, 1272, 670,  1356, 670,  372,
       1511, 1970, 8,    20,   45,   7005, 1293, 655,  2681, 7824, 779,  7746,
       758,  1417, 734,  5124, 1167, 4879, 815,  1327, 2793, 4726, 3899, 1000,
       8193, 83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   45,   45,   248},
      {8,    7406, 5978, 1601, 3487, 6693, 3893, 2603, 1100, 612,  7403, 4584,
       8,    20,   45,   83,   45,   299,  2867, 1197, 230,  2071, 2283, 6497,
       7683, 1084, 4357, 492,  1265, 1835, 2021, 989,  2929, 2159, 1374, 7005,
       8193, 83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,   83,
       83,   83,   83,   83,   83,   45,   45,   248}};

  if (!mel_code_vectors_match(sequences, target_sequences)) {
    std::cout << "token sequence mismatch" << std::endl;
    exit(1);
  }

  if (!latent_vectors_match(trimmed_latents, target_trimmed_latents)) {
    std::cout << "trimmed latent mismatch" << std::endl;
    exit(1);
  }

  std::cout << "AUTOREGRESSIVE TEST SUCCESS!" << std::endl;
}

void test_diffusion() {

  // load rng state
  // https://stackoverflow.com/a/42386852 used for reference
  std::ifstream fin("../assets/test_diffusion_seed.bin");
  fin >> generator;
  fin.close();
  std::ifstream fin2("../assets/test_diffusion_normal_distribution.bin");
  fin2 >> distribution;
  fin2.close();

  std::vector<float> diffusion_input =
      load_f32_vector("../assets/diffusion_input.bin", 44032 * sizeof(float));
  std::vector<float> target_mel =
      load_f32_vector("../assets/target_mel.bin", 18700 * sizeof(float));
  std::vector<float> mel = diffusion(diffusion_input);
  if (!vectors_match(target_mel, mel)) {
    std::cout << "mel mismatch" << std::endl;
    exit(1);
  }

  std::cout << "DIFFUSION TEST SUCCESS!" << std::endl;
}

void test_vocoder() {

  std::vector<float> vocoder_input =
      load_f32_vector("../assets/target_mel.bin", 18700 * sizeof(float));

  std::vector<float> target_audio =
      load_f32_vector("../assets/target_audio.bin", 50426 * sizeof(float));

  std::vector<float> audio = vocoder(vocoder_input);
  if (!vectors_match(target_audio, audio)) {
    std::cout << "mel mismatch" << std::endl;
    exit(1);
  }

  std::cout << "VOCODER TEST SUCCESS!" << std::endl;
}

// clang-format off

/*

 ███╗   ███╗ █████╗ ██╗███╗   ██╗
 ████╗ ████║██╔══██╗██║████╗  ██║
 ██╔████╔██║███████║██║██╔██╗ ██║
 ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
 ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
 ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝


*/

// clang-format on

int main(int argc, char **argv) {

  std::string defaultMessage = "this is a test message.";
  std::string defaultVoicePath = "../models/mol.bin";
  std::string defaultOutputPath = "./output.wav";
  bool defaultTimeResult = false;
  std::string message = defaultMessage;
  std::string voicePath = defaultVoicePath;
  std::string outputPath = defaultOutputPath;
  bool timeResult = defaultTimeResult;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--voice") {
      voicePath = argv[i + 1];
    } else if (std::string(argv[i]) == "--message") {
      message = argv[i + 1];
    } else if (std::string(argv[i]) == "--output") {
      outputPath = argv[i + 1];
    } else if (std::string(argv[i]) == "--seed") {
      generator.seed(std::stoi(argv[i + 1]));
    } else if (std::string(argv[i]) == "--time") {
      timeResult = true;
    }
  }

  gpt_vocab vocab;
  gpt_vocab_init("../models/tokenizer.json", vocab);

  // test_autoregressive();
  // test_diffusion();
  // test_vocoder();
  // std::cout << "all tests succeeded!" << std::endl;
  // exit(0);

  auto start = std::chrono::high_resolution_clock::now();

  replaceAll(message, " ", "[SPACE]");

  std::vector<gpt_vocab::id> pre = ::parse_tokens_from_string("255", ',');
  std::vector<gpt_vocab::id> post = ::parse_tokens_from_string("0", ',');

  std::vector<gpt_vocab::id> tokens = ::gpt_tokenize(vocab, message);

  tokens.insert(tokens.begin(), pre.begin(), pre.end());
  tokens.insert(tokens.end(), post.begin(), post.end());

  std::pair<std::vector<std::vector<float>>, std::vector<std::vector<int>>>
      autoregressive_result = autoregressive(tokens, voicePath, 1);

  std::vector<std::vector<float>> trimmed_latents = autoregressive_result.first;
  std::vector<std::vector<int>> sequences = autoregressive_result.second;

  std::vector<float> mel = diffusion(trimmed_latents[0]);

  std::vector<float> audio = vocoder(mel);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  // save_f32_vector("./target_audio.bin", audio);

  if (timeResult) {
    std::cout << "Time taken: " << duration_ms << " milliseconds" << std::endl;
  }

  writeWav(outputPath.c_str(), audio, 24000);
  std::cout << std::endl << std::endl;
  std::cout << "Please consider donating :^)" << std::endl
            << "https://ko-fi.com/johnbalis" << std::endl;

  return 0;
}
