# random assorted nonsense

- apparently the model is relatively fine with replacing input tokens with `<pad>`
  even without `short_ctx_dropout_p > 0`
- refuses to converge on Google's TPUs for some reason
  - not even with `torch_xla._XLAC._xla_set_use_full_mat_mul_precision(True)`
- refuses to converge on bfloat16 for some reason
