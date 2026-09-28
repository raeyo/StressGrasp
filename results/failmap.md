# 실패 조건 지도

> 판정 없음. 조건별 SR 과 clean 대비 낙폭만. clean 은 각 (정책,손,물체셋) 자기 자신 기준.

| 정책 | 손 | 물체셋 | 모드 | 조건 | SR | ΔSR |
|---|---|---|---|---|---|---|
| fr3_dclaw | fr3_dclaw_gripper | ours_S | teacher | clean | 0.2621 | **+0.0000** |
| fr3_panda_gripper | fr3_panda_gripper | ours_L | teacher | clean | 0.5574 | **+0.0000** |
| fr3_panda_gripper | fr3_panda_gripper | ours_M | teacher | clean | 0.8333 | **+0.0000** |
| fr3_panda_gripper | fr3_panda_gripper | ours_S | teacher | clean | 0.8667 | **+0.0000** |
| fr3_panda_gripper | fr3_panda_gripper | ours_S | teacher | 물체변위 4cm (파지 전) | 0.0803 | **-0.7864** |
| fr3_shadow | fr3_shadow | ours_S | teacher | clean | 0.5318 | **+0.0000** |
| inspire | default | ours_L | teacher | clean | 0.9353 | **+0.0000** |
| inspire | default | ours_M | teacher | clean | 0.8955 | **+0.0000** |
| inspire | default | ours_S | teacher | clean | 0.7621 | **+0.0000** |
| inspire | default | ours_S | teacher | 질량 x20 (0.62kg) | 0.7136 | **-0.0485** |
| inspire | default | ours_S | teacher | 질량 x24 (0.74kg) | 0.6333 | **-0.1288** |
| inspire | default | ours_S | teacher | 질량 x48 (1.48kg) | 0.3470 | **-0.4152** |
| inspire | default | ours_S | teacher | 질량 x64 (1.98kg) | 0.2258 | **-0.5364** |
| inspire | default | ours_S | teacher | 물체변위 1cm (파지 전) | 0.7424 | **-0.0197** |
| inspire | default | ours_S | teacher | 물체변위 2cm (파지 전) | 0.7379 | **-0.0242** |
| inspire | default | ours_S | teacher | 물체변위 4cm (파지 전) | 0.6121 | **-0.1500** |
| inspire | default | ours_S | teacher | 물체변위 8cm (파지 전) | 0.1470 | **-0.6152** |
| inspire | default | ours_S | teacher | 물체변위 12cm (파지 전) | 0.0303 | **-0.7318** |
| inspire | default | ours_S | teacher | 물체변위 16cm (파지 전) | 0.0045 | **-0.7576** |
| shadow | shadow_simple | ours_L | teacher | clean | 0.9206 | **+0.0000** |
| shadow | shadow_simple | ours_M | teacher | clean | 0.8576 | **+0.0000** |
| shadow | shadow_simple | ours_S | teacher | clean | 0.6000 | **+0.0000** |
| shadow | shadow_simple | ours_S | teacher | 물체변위 4cm (파지 전) | 0.1818 | **-0.4182** |
| ur5_allegro | ur5_allegro | ours_L | teacher | clean | 0.9206 | **+0.0000** |
| ur5_allegro | ur5_allegro | ours_M | teacher | clean | 0.8485 | **+0.0000** |
| ur5_allegro | ur5_allegro | ours_S | teacher | clean | 0.5348 | **+0.0000** |
| ur5_svh | ur5_svh | ours_S | teacher | clean | 0.7106 | **+0.0000** |
