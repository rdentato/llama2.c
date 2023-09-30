/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#define VRGCLI
#include "vrg.h"

#include "tensor.h"

#define Embedding(t)   tensor(transformer.weights.token_embedding_table + (t) * llm_dim , llm_dim)

#define W1(l)      tensor(transformer.weights.w1                    , llm_hidden_dim, llm_dim       , l)
#define W3(l)      tensor(transformer.weights.w3                    , llm_hidden_dim, llm_dim       , l)
#define W2(l)      tensor(transformer.weights.w2                    , llm_dim       , llm_hidden_dim, l)
#define Wq(l)      tensor(transformer.weights.wq                    , llm_dim       , llm_dim       , l)
#define Wk(l)      tensor(transformer.weights.wk                    , llm_kv_dim    , llm_dim       , l)
#define Wv(l)      tensor(transformer.weights.wv                    , llm_kv_dim    , llm_dim       , l)
#define Wo(l)      tensor(transformer.weights.wo                    , llm_dim       , llm_dim       , l)
#define Wcls()     tensor(transformer.weights.wcls                  , llm_vocab_size, llm_dim          )
#define RMSatt(l)  tensor(transformer.weights.rms_att_weight        , 1             , llm_dim       , l)
#define RMSffn(l)  tensor(transformer.weights.rms_ffn_weight        , 1             , llm_dim       , l)
#define RMSfinal() tensor(transformer.weights.rms_final_weight      , 1             , llm_dim          )

#define X()      tensor(transformer.state.x,llm_dim)
#define Xb2()    tensor(transformer.state.xb2,llm_dim)
#define Hb()     tensor(transformer.state.hb,llm_hidden_dim)
#define Hb2()    tensor(transformer.state.hb2,llm_hidden_dim)
#define K()      tensor(transformer.state.k,llm_kv_dim)
#define V()      tensor(transformer.state.v,llm_kv_dim)
#define Att(h)   tensor(transformer.state.att + (h) * llm_seq_len, llm_seq_len)

#define Xb(...)  vrg(Xb, __VA_ARGS__)
#define Xb01()   tensor(transformer.state.xb,llm_dim)
#define Xb_1(h)  tensor(transformer.state.xb+ (h) * llm_head_size, llm_head_size)


#define Q(...)   vrg(Q,__VA_ARGS__)
#define Q01()    tensor(transformer.state.q,llm_dim)
#define Q_1(h)   tensor(transformer.state.q + (h) * llm_head_size, 1, llm_head_size)

#define Logits() tensor(transformer.state.logits,llm_vocab_size)

#define Kcache(...) vrg(Kcache,__VA_ARGS__)
#define Kcache_2(l,p)   tensor(transformer.state.key_cache + ((l) * llm_seq_len * llm_kv_dim) + (p) * llm_kv_dim, llm_kv_dim)
#define Kcache_3(l,p,h) tensor(transformer.state.key_cache + ((l) * llm_seq_len * llm_kv_dim) + (p) * llm_kv_dim + ((h) / llm_kv_mul) * llm_head_size, llm_head_size)

#define Vcache(...) vrg(Vcache,__VA_ARGS__)
#define Vcache_2(l,p)   tensor(transformer.state.value_cache + ((l) * llm_seq_len * llm_kv_dim) + (p) * llm_kv_dim, llm_kv_dim)
#define Vcache_3(l,p,h) tensor(transformer.state.value_cache + ((l) * llm_seq_len * llm_kv_dim) + (p) * llm_kv_dim + ((h) / llm_kv_mul) * llm_head_size, llm_head_size)


// ----------------------------------------------------------------------------
// Transformer model
    int llm_dim;
    int llm_hidden_dim;
    int llm_n_layers;
    int llm_n_heads;
    int llm_n_kv_heads;
    int llm_vocab_size;
    int llm_seq_len;
    int llm_kv_dim;
    int llm_kv_mul;
    int llm_head_size;
    int llm_shared_weights;
    int llm_header_size = 7 * sizeof(int);
    ssize_t llm_file_size;

    char *llm_checkpoint_path;
    char *llm_tokenizer_path = "tokenizer.bin";


typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

Transformer transformer;

int load_config(char *checkpoint) 
{ 
    llm_checkpoint_path = checkpoint;
    FILE *f = fopen(llm_checkpoint_path, "rb");
    if (!f) { fprintf(stderr, "Couldn't open file %s\n", llm_checkpoint_path); exit(EXIT_FAILURE); }

    //if (fread(c, llm_header_size, 1, f) != 1) { exit(EXIT_FAILURE); } 
    if (fread(&(llm_dim       ), sizeof(int), 1, f) != 1) { exit(EXIT_FAILURE); }; // transformer dimension
    if (fread(&(llm_hidden_dim), sizeof(int), 1, f) != 1) { exit(EXIT_FAILURE); }; // for ffn layers
    if (fread(&(llm_n_layers  ), sizeof(int), 1, f) != 1) { exit(EXIT_FAILURE); }; // number of layers
    if (fread(&(llm_n_heads   ), sizeof(int), 1, f) != 1) { exit(EXIT_FAILURE); }; // number of query heads
    if (fread(&(llm_n_kv_heads), sizeof(int), 1, f) != 1) { exit(EXIT_FAILURE); }; // number of key/value heads (can be < query heads because of multiquery)
    if (fread(&(llm_vocab_size), sizeof(int), 1, f) != 1) { exit(EXIT_FAILURE); }; // vocabulary size, usually 256 (byte-level)
    if (fread(&(llm_seq_len   ), sizeof(int), 1, f) != 1) { exit(EXIT_FAILURE); }; // max sequence length

    llm_shared_weights = llm_vocab_size > 0 ? 1 : 0; 
    llm_vocab_size     = abs(llm_vocab_size); 
    llm_kv_dim         = (llm_dim * llm_n_kv_heads) / llm_n_heads; 
    llm_kv_mul         = llm_n_heads / llm_n_kv_heads; 
    llm_head_size      = llm_dim / llm_n_heads; 

    // figure out the file size
    fseek(f, 0, SEEK_END); // move file pointer to end of file
    llm_file_size = ftell(f); // get the file size, in bytes
    fclose(f);

    return llm_header_size;
}

void malloc_run_state(RunState* s) {
    // we calloc instead of malloc to keep valgrind happy
    s->x = calloc(llm_dim, sizeof(float));
    s->xb = calloc(llm_dim, sizeof(float));
    s->xb2 = calloc(llm_dim, sizeof(float));
    s->hb = calloc(llm_hidden_dim, sizeof(float));
    s->hb2 = calloc(llm_hidden_dim, sizeof(float));
    s->q = calloc(llm_dim, sizeof(float));
    s->k = calloc(llm_kv_dim, sizeof(float));
    s->v = calloc(llm_kv_dim, sizeof(float));
    s->att = calloc(llm_n_heads * llm_seq_len, sizeof(float));
    s->logits = calloc(llm_vocab_size, sizeof(float));
    s->key_cache = calloc(llm_n_layers * llm_seq_len * llm_kv_dim, sizeof(float));
    s->value_cache = calloc(llm_n_layers * llm_seq_len * llm_kv_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void build_transformer(Transformer *t) {
    // memory map the Transformer weights into the data pointer
    t->fd = open(llm_checkpoint_path, O_RDONLY); // open in read only mode
    if (t->fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    t->data = mmap(NULL, llm_file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
    if (t->data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float *ptr = t->data + llm_header_size/sizeof(float);

    TransformerWeights *w = &(transformer.weights);
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = llm_n_layers;
    w->token_embedding_table = ptr;    ptr += llm_vocab_size * llm_dim;
    w->rms_att_weight = ptr;           ptr += n_layers * llm_dim;
    w->wq = ptr;                       ptr += n_layers * llm_dim * (llm_n_heads * llm_head_size);
    w->wk = ptr;                       ptr += n_layers * llm_dim * (llm_n_kv_heads * llm_head_size);
    w->wv = ptr;                       ptr += n_layers * llm_dim * (llm_n_kv_heads * llm_head_size);
    w->wo = ptr;                       ptr += n_layers * (llm_n_heads * llm_head_size) * llm_dim;
    w->rms_ffn_weight = ptr;           ptr += n_layers * llm_dim;
    w->w1 = ptr;                       ptr += n_layers * llm_dim * llm_hidden_dim;
    w->w2 = ptr;                       ptr += n_layers * llm_hidden_dim * llm_dim;
    w->w3 = ptr;                       ptr += n_layers * llm_dim * llm_hidden_dim;
    w->rms_final_weight = ptr;         ptr += llm_dim;
                                       ptr += llm_seq_len * llm_head_size / 2; // skip what used to be freq_cis_real (for RoPE)
                                       ptr += llm_seq_len * llm_head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = llm_shared_weights ? w->token_embedding_table : ptr;

    // allocate the RunState buffers
    malloc_run_state(&t->state);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, llm_file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(tensor_t oo, tensor_t xx, tensor_t ww) {
    float* o = tensor_values(oo);
    float* x = tensor_values(xx);
    float* weight = tensor_values(ww);
    int size = tensor_ncols(ww);
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(tensor_t xx, int size) {
    float* x = tensor_values(xx);
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(tensor_t Xout, tensor_t W, tensor_t Xin) {
    int d = tensor_nrows(W);
    int n = tensor_ncols(W);

    // fprintf(stderr,"Xout[%d,%d] Xin[%d,%d] W[%d,%d]\n", tensor_nrows(Xout), tensor_ncols(Xout) , tensor_nrows(Xin), tensor_ncols(Xin), tensor_nrows(W), tensor_ncols(W));
    //assert(tensor_nrows(W) == tensor_nrows(Xout));

    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += tensor_get(W,i,j) * tensor_get(Xin,j);
        }
        tensor_set(Xout,val,i);
    }
}

void RoPE(int pos)
{
    for (int i = 0; i < llm_dim; i+=2) {
        int head_dim = i % llm_head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)llm_head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < llm_kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? tensor_values(Q()) : tensor_values(K()); // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}

float* forward(int token, int pos) {

    RunState* s = &(transformer.state);

    // copy the token embedding into x
    tensor_copy(X(), Embedding(token));

    // forward all the layers
    for(unsigned long long l = 0; l < llm_n_layers; l++) {

        // attention rmsnorm
        rmsnorm(Xb(), X(), RMSatt(l));

        // qkv matmuls for this position
        matmul(Q(), Wq(l), Xb());
        matmul(K(), Wk(l), Xb());
        matmul(V(), Wv(l), Xb());

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        RoPE(pos);

        // save key,value at this time step (pos) to our kv cache
        tensor_copy(Kcache(l,pos), K());
        tensor_copy(Vcache(l,pos), V());

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < llm_n_heads; h++) {

            // attention scores for this head
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float score;
                score = tensor_dot(Q(h), Kcache(l,t,h));
                score /= sqrtf(llm_head_size);
                tensor_set(Att(h),score,t);
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(Att(h), pos + 1);

            // weighted sum of the values, store back into xb
            tensor_zero(Xb(h));

            // memset(xb, 0, llm_head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the attention weight for this timestep
                float a = tensor_get(Att(h),t);

                // accumulate the weighted value into xb
                tensor_add(Xb(h), Xb(h), Vcache(l,t,h), a); // Xb = Xb + a V
            }
        }

        // final matmul to get the output of the attention
        matmul(Xb2(), Wo(l), Xb());

        // residual connection back into x
        tensor_add(X(), Xb2());
 
        // ffn rmsnorm
        rmsnorm(Xb(), X(), RMSffn(l));

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(Hb() , W1(l), Xb());
        matmul(Hb2(), W3(l), Xb());

        // SwiGLU non-linearity
        for (int i = 0; i < llm_hidden_dim; i++) {
            float val = tensor_get(Hb(),i);
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= tensor_get(Hb2(),i);
            tensor_set(Hb(),val,i);
        }

        // final matmul to get the output of the ffn
        matmul(Xb(), W2(l), Hb());

        // residual connection
        tensor_add(X(), X(), Xb());
    }

    // final rmsnorm
    rmsnorm(X(), X(), RMSfinal());

    // classifier into logits
    matmul(Logits(), Wcls(), X());
    return tensor_values(Logits());
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = llm_vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(llm_vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(llm_vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(llm_tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", llm_tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < llm_vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(Logits(), sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    vrgcli("Example: run model.bin -n 256 -i \"Once upon a time\"") {
       vrgarg("checkpoint\tModel checkpoint") 
           checkpoint_path = vrgarg;

       vrgarg("-t, --temp <float>\ttemperature in [0,inf], default 1.0")
           temperature = atof(vrgarg); 

       vrgarg("-p, --top-p <float>\tp value in top-p (nucleus) sampling in [0,1] default 0.9")
           topp = atof(vrgarg);

       vrgarg("-s, --seed <int>\trandom seed, default time(NULL)")
           rng_seed = atoi(vrgarg); 

       vrgarg("-n, --steps <int>\tnumber of steps to run for, default 256. 0 = max_seq_len")
           steps = atoi(vrgarg);

       vrgarg("-i, --input <string>\tinput prompt")
           prompt = vrgarg;

       vrgarg("--prompt <string>\tinput prompt")
           prompt = vrgarg;

       vrgarg("-z, --tokenizer <string>\toptional path to custom tokenizer")
           llm_tokenizer_path = vrgarg;

       vrgarg("-h, --help\tPrints this help") vrgusage();

       vrgarg() vrgusage("Unexpected argument '%s'\n",vrgarg);

    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    load_config(checkpoint_path);
    if (steps == 0 || steps > llm_seq_len) steps = llm_seq_len; // ovrerride to ~max length

    // build the Transformer via the model .bin file
    build_transformer(&transformer);

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, llm_vocab_size, temperature, topp, rng_seed);

    // run!
    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
