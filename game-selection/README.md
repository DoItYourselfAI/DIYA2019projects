# game-selection

어떤 게임으로 대회를 열 지 정해 봅시다.

의견은 이 파일을 수정하여 커밋하시거나, 리포지토리 위키를 이용해서 자유롭게 써주세요.

## 기준
- BizHawk 에뮬레이터에서 작동할 것 
- 16비트 이하 게임 권장 (메모리 값을 보고 의미를 찾아내기 쉬울듯, 아래는 8, 16비트 콘솔들)
    - [메가 드라이브](https://ko.wikipedia.org/wiki/%EB%A9%94%EA%B0%80_%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C)[^1]
    - [NES(패밀리 컴퓨터)](https://ko.wikipedia.org/wiki/%ED%8C%A8%EB%B0%80%EB%A6%AC_%EC%BB%B4%ED%93%A8%ED%84%B0)[^2]
    - [SNES(슈퍼 패미컴)](https://ko.wikipedia.org/wiki/%EC%8A%88%ED%8D%BC_%ED%8C%A8%EB%AF%B8%EC%BB%B4)[^3]
    - [게임보이](https://ko.wikipedia.org/wiki/%EA%B2%8C%EC%9E%84%EB%B3%B4%EC%9D%B4)[^4]
    - (추가바람)
- PvP 대전게임으로 두 명 이상의 플레이어가 겨루는 형식
- 게임에 내장된 AI가 있어서 RL 모델 학습이 용이할 것
- 학습된 에이전트 모델이 Github등 공개 저장소에 돌아다니지 않는 것


## 후보 (추가하고 merge해주세요)

### [Tecmo World Cup Soccer](https://www.gamespot.com/tecmo-world-cup-soccer/) (JP:1990/EU:1991)

![Tecmo World Cup Soccer](http://www.mobygames.com/images/shots/l/548366-tecmo-world-cup-soccer-nes-screenshot-ball-is-high-up.png)

- 플랫폼: NES
- 특징
    - 2P 지원
    - (추가바람)
- [플레이 영상](https://www.youtube.com/watch?v=Oz6GzDloQrg)

### [Super bomberman 2](https://www.gamespot.com/super-bomberman-2/) (JP:1994/NA:1994/EU:1995)

<img src="https://iv1.lisimg.com/image/265909/638full-super-bomberman-2-screenshot.jpg" width="384" height="256">
<img src="https://i.pinimg.com/originals/e5/6c/bf/e56cbf0a95af0748a3f921f25d09be2e.jpg" width="200" height="256">`

- 플랫폼: SNES
- 특징
    - 4P 지원[^5]
    - 게임이 간단, 4배속도 무리가 없음
- [플레이 영상](https://www.youtube.com/watch?v=Ae-sjJfg6fQ)

### [Mortal Kombat 1](https://www.gamespot.com/mortal-kombat/) (NA:1992)

<img src="https://images.gog.com/afa3762f2882f50d6d9fac5127e7714db86084b9a5b98ed9e57c4ab42c50d907_product_card_v2_thumbnail_542.jpg" width="384" height="256">
<img src="https://images.gog.com/288e2345eb84d9a04301e739ad653b1ab89c858f0c73e5827a90d2f8fb072bb1_product_card_v2_thumbnail_542.jpg" width="200" height="256">`

- 플랫폼: SNES
- 특징
    - 2P 지원
    - 게임이 간단, 보상 설계 직관적(?)
    - BizHawk 에뮬레이터 학습 가능 https://arxiv.org/pdf/1611.02205.pdf (공개된 코드 없음)

- [플레이 영상](https://www.youtube.com/watch?v=7F82XAvCuVc)

### [Mario Kart 64](https://www.gamespot.com/mario-kart-64/) (JP:1996/NA:1997/EU:1997)

<img src="https://www.retrogames.cc/screenshot/59/w/32603_b7665f16d2af67e00c86921c45634c919af75f51.png" width="384" height="256">
<img src="https://compote.slate.com/images/1983c4e4-a18e-43b8-b456-244003f50521.jpg" width="200" height="256">`

- 플랫폼: Nintendo 64
- 특징
    - 멀티플레이어 지원 (최대 4P)
    - 다양한 모드 지원 (Time-Trial, Grand Prix, Battle)
    - BizHawk 에뮬레이터 학습 가능 http://cs231n.stanford.edu/reports/2017/pdfs/624.pdf
    - 공개된 코드가 있으나, 1P Time-Trial 모드에서만 학습

- [플레이 영상](https://youtu.be/1_9B7OiC_BI)

---



[^1]: [Zilog Z80](https://www.zilog.com/manage_directlink.php?filepath=docs/z80/um0080&extn=.pdf) , [Motorola 68000](https://www.nxp.com/files-static/archives/doc/ref_manual/M68000PRM.pdf)
[^2]: [ricoh 2A03](https://en.wikibooks.org/wiki/NES_Programming)
[^3]: [ricoh 5A22](https://en.wikibooks.org/wiki/Super_NES_Programming/65c816_reference)
[^4]: [LR 35902](http://marc.rawer.de/Gameboy/Docs/GBCPUman.pdf)
[^5]: core를 bsnes로 설정할 경우 사용 가능
