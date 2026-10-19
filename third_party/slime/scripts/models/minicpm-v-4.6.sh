MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 8
   --num-query-groups 2
   --kv-channels 256
   --num-layers 24
   --hidden-size 1024
   --ffn-hidden-size 3584
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-10000000}"
   --swiglu

   --vocab-size 248320

   --attention-output-gate
   --qwen-gdn-backend "${QWEN_GDN_BACKEND:-fla}"
   --loss-mask-type qwen3_5
)
