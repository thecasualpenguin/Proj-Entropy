# Third-party notices

## ConvTran

`NativeConvTran` adapts the architecture described and implemented by Navid
Fouladi's [ConvTran](https://github.com/Navidfoumani/ConvTran) project,
including tAPE and post-softmax eRPE attention.

- Upstream repository: https://github.com/Navidfoumani/ConvTran
- Upstream license: MIT (reviewed 2026-10-11)

No upstream source files were copied into this WPP classifier. The local module
is a clean, minimal implementation specialized for masked, variable-row WPP
inputs; its convolutional tokenizer, data contract, and masking logic are
local work. Keep this notice and the upstream MIT attribution when modifying
or redistributing the adaptation.
