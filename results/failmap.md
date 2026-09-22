# 실패 조건 지도

> 판정 없음. 조건별 SR 과 clean 대비 낙폭만. clean 은 각 (정책,손,물체셋) 자기 자신 기준.

| 정책 | 손 | 물체셋 | 모드 | 조건 | SR | ΔSR |
|---|---|---|---|---|---|---|
| fr3_shadow | fr3_shadow | ours_S | teacher | clean | 0.5318 | **+0.0000** |
| inspire | default | ours_S | teacher | clean | 0.7621 | **+0.0000** |
| inspire | default | ours_S | teacher | 물체변위 1cm (파지 전) | 0.7712 | **-0.0091** |
| inspire | default | ours_S | teacher | 물체변위 2cm (파지 전) | 0.7652 | **-0.0030** |
| inspire | default | ours_S | teacher | 물체변위 4cm (파지 전) | 0.6015 | **+0.1606** |
| inspire | default | ours_S | teacher | 물체변위 8cm (파지 전) | 0.1470 | **+0.6152** |
| shadow | shadow_simple | ours_S | teacher | clean | 0.6000 | **+0.0000** |
