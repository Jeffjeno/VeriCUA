MODEL_ARGS=(
   --swiglu
   --num-layers 32
   --hidden-size 2560
   --ffn-hidden-size 10240
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 2
   --max-position-embeddings 32768
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-10000}"
   --vocab-size 73448
   --kv-channels 128
   --untie-embeddings-and-output-weights
)
