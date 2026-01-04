Method: HeightmapMin, LSAH, MACS, RANDOM, OnlineBPH, DBL, BR, SDFPack


```bash
python main.py --method HeightmapMin
```
debug:

main.py(122行)  中packer.pack_box() 调用不同的method决策

packer/app.py(1039行) 为每种method的具体调用逻辑

其中得到的action与PCT逻辑相同, 直接运行main可以看到在仿真环境中具体的摆放效果, 便于调试
