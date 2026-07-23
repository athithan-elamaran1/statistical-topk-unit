# Verification

cocotb testbench verifying `tt_um_statistical_topk` bit-for-bit against the
Python golden model in `../sim/topk_model.py`. Eight tests, ~4,900 samples:
reset state, out_valid timing, Gaussian-stream bit-exactness, warmup boundary,
live reconfiguration, extreme inputs, distribution shift, busy-edge protocol.

```bash
pip install -r requirements.txt   # cocotb + pytest
make                              # RTL sim (icarus)
make GATES=yes                    # gate-level sim (CI provides the netlist)
```

Only TT-level pins are observed, so the identical suite runs against the
post-layout gate-level netlist.

The serial multiply engine also has a standalone self-checking testbench
(directed corners + 500 random products), independent of cocotb:

```bash
iverilog -g2005 -o tb_mac tb_mac.v ../src/serial_mac.v && vvp tb_mac
```
