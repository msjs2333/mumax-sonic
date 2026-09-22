# P4 收尾：固定帧对照试听

使用 P4a、P4b、P4c 已完成的数据，比较 FM 活动、AFM Néel 活动、带内同相/反相和带外强度；另生成正、负、净零双纹理三个合成拓扑片段。合成纹理只用于声部与空间覆盖对照，不冒充真实求解器拓扑态。

先按各案例说明生成数据，再准备一个新目录（路径替换为自己的运行结果）：

```powershell
python scripts/prepare_p4_listening.py local/p4-listening --fm local/sp4-field1/replay.json --afm local/afm-small/neel.json --band-root local/band-small
python scripts/replay_listening_check.py local/p4-listening/playlist.json --report local/p4-listening/preview.json
python scripts/replay_listening_check.py local/p4-listening/playlist.json --play --seconds 3 --report local/p4-listening/device-play.json
```

第一条不运行求解器。第二条只准备和检查声音参数，不打开设备；第三条使用 OpenAL 设备播放 8 段，每段 3 秒，间隔静音。已有报告不覆盖，重复运行需换报告文件名。终端显示当前片段，Ctrl+C 可停止。设备音量使用自己舒适的设置；比较空间方位时使用耳机。

所有文件读取、物理观察和聚合先完成，再打开音频设备。每段保持一个选定帧的场景；Activity 使用真实前驱，Band 使用截止该帧的完整物理窗口。刷新声音控制不会推进物理时间。这适合比较声部和固定尺度强度，不验证动态拖动、跟踪或实时延迟。

| 顺序 | 片段 | 听觉比较重点 |
| --- | --- | --- |
| 1–2 | 真实 FM / AFM 活动 | 是否有可辨、舒适的声音；两种参考尺度不同，不能跨案例用音量直接比较物理大小 |
| 3–4 | 真实带内同相 / 反相 | 应保持接近的强度，不能因横向空间均值抵消而静音 |
| 5 | 真实带外 | 相同频带与参考尺度下应明显弱于带内；保留原始残余，不自动拉满 |
| 6–7 | 合成拓扑正 / 负 | 是否能分辨两个声部，不以负振幅当负号 |
| 8 | 合成净零双纹理 | 是否能听见两种贡献，而非只听净拓扑荷 |

播放使用固定总增益缩放 0.2，不对各片段重新归一化。FM 活动参考为 10¹⁰ rad/s，AFM 为 10¹¹ rad/s，三组频带共用 0.01，拓扑共用 1。默认空间预算为 8 路。列表中可更改选帧和参考尺度；数据未就绪、缺帧或物理观察无效时拒绝播放，不伪装为物理零。

请记录实际耳机/输出设备、系统音量、哪些片段听不清、同相/反相强度是否相近，以及正负声部能否区分。自动报告永远保留 `human_listening_verified=false`，不代填主观结论；当前音色先保持一致，收到听觉反馈再调整。
