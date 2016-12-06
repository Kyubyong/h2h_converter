# h2h Converter
Convert Sino-Korean words written in <b>H</b>angul to Chinese characters, which is called <b>H</b>anja in Korean using neural networks

## Requirements
  * numpy >= 1.11.1
  * sugartensor == 0.0.1.8 (Check [this repository](https://github.com/buriburisuri/sugartensor) for installing sugartensor)
  * + Your Own Corpus
	
## Research Question
Can we convert Sino-Korean words written in Hangul, the Korean alphabet, to Chinese character correctly using neural networks?

## Background
  * Around 2/3 of Korean words are Sino-Korean.
  * The official script of Korean is Hangul, but Chinese characters, Hanja in Korean, are still used.
  * Transcripting Hanja to Hangul is trivial because most of Hanja have a single equivalent of Hangul. However, the reverse is not.
  * A Hangul-to-Hanja conveter demo, Utagger, is avaiable [here](http://203.250.77.242:5900/utagger/).
  * Neural networks, concretely CNNs, can be applied to this task.

## Main Idea
We use dilated convolutional neural networks instead of biRNN. It turns out that the former is more powerful and much faster than the latter.

## Data
We built a hangul-hanja parallel corpus for the research, which is neither big enough nor excellent in quality. It looks like the following.

나는 오늘 학교에 간다 <b>[Tab]<b/> 나는 오늘 學校에 간다

## Model Architecture

Two blocks of convolutional neural networks of exponentially increased holes. (Check [Kalchbrenner et. al.](https://arxiv.org/pdf/1610.10099v1.pdf) )

And its implementation code is borrowed from Namju's github repository. (Check Namju's [ByteNet](https://github.com/buriburisuri/ByteNet) )

## Folder and file instructions
  * prepro.py: Preprocessing. Make your own corpus at `corpus/corpus.tsv`. Returns `data/X.npy`, `data/Y.npy`, and `data/charmaps.pkl`. You can adjust hyperparameter in this file if you want.
  * train.py: Training. By default, it creates log files and model files at `asset/train/log` and `asset/train/ckpt` respectively.
  * run.py: Running/Testing. Read `data/input.txt` and write the results to `data/output.txt`. You can see our sample input file and its results.
  * model-019-1239684: Pretrained model parameters. Can be download [here](https://drive.google.com/open?id=0B5M-ed49qMsDQ1dEYXF3cTVNM1E).

## Results
After having seen 19 epochs, we test some sample sentences. Here are some snippets of the test results.

▌Input: 하루 새 공기의 느낌이 확 달라졌습니다.<br/> 
▌Output: 하루 새 空氣의 느낌이 확 달라졌습니다.

▌Input: 밤새 찬바람이 불면서 미세먼지는 물러갔지만, 겨울 추위가 찾아왔는데요, <br/> 
▌Output: 밤새 찬바람이 불면서 微細먼지는 물러갔지만, 겨울 추위가 찾아왔는데요,

▌Input: 현재 서울 등 수도권 쪽으로는 한파주의보도 내려진 상태입니다.<br/> 
▌Output: 현재 서울 等 首都권 쪽으로는 韓波主義보도 내려진 狀態입니다.

▌Input: 현재 서울의 기온 영하 3.9 도, 철원 영하 5.1 도, 남부 내륙도 영하로 뚝 떨어져 있고요, <br/> 
▌Output: 현재 서울의 氣溫 영하 3.9 도, 鐵圓 영하 5.1 도, 南部 內陸도 영하로 뚝 떨어져 있고요,

▌Input: 초속 4m 안팎의 찬바람이 불면서 현재 서울의 체감온도는 영하 10도 가까이 내려가 있는 상태입니다.<br/> 
▌Output: 初速 4m 안팎의 찬바람이 불면서 현재 서울의 體感온도는 영하 10도 가까이 내려가 있는 狀態입니다.

▌Input: 동해안 쪽으로는 건조특보가 계속되고 있어서 화재 예방에 유의를 하셔야겠고요, <br/> 
▌Output: 東海岸 쪽으로는 乾組특보가 繼續되고 있어서 火災 豫防에 유의를 하셔야겠고요, 

▌Input: 내일은 중부 지방을 중심으로 1cm 안팎의 눈이 조금 내리겠습니다.<br/> 
▌Output: 내일은 中部 地方을 中心으로 1cm 안팎의 눈이 조금 내리겠습니다.

▌Input: 오늘 하늘은 종일 맑겠지만, 찬바람이 불면서 춥겠습니다.<br/> 
▌Output: 오늘 하늘은 終日 맑겠지만, 찬바람이 불면서 춥겠습니다.

▌Input: 낮 기온은 서울 3도, 청주 5도에 머무는 데다가, 체감온도는 더 낮겠습니다.<br/> 
▌Output: 낮 氣溫은 서울 3도, 靑紬 5도에 머무는 데다가, 體感온도는 더 낮겠습니다.

▌Input: 내일은 새벽부터 낮 사이 중부 지방을 중심으로 비나 눈이 조금 예상됩니다. <br/> 
▌Output: 내일은 새벽부터 낮 사이 中部 地方을 中心으로 비나 눈이 조금 豫想됩니다. 

## Conclusion
Convolutional neural networks are effective in hangul-to-hanja conversion. However, we need more data, and a bigger model!





