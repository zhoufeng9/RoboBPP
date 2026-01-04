This code package contains the training of the AR2L algorithm. Please do not distribute it.
Before executing the training process, please ensure that the necessary requirements have been installed.

1. Before executing the training process, please ensure that the necessary requirements have been installed.
pip install -r requirements.txt

2. train ExactAR2L in the discrete setting.
python main.py --num-box 80 --num-next-box 20 --num-candidate-action 120 --alpha 1

3. train ExactAR2L in the continuous setting.
python main.py \
--container-size 1.0 \
--num-box 100 \
--num-next-box 20 \
--num-candidate-action 120 \
--continuous \
--sample-from-distribution \
--sample-left-bound 0.1 \
--sample-right-bound 0.5 \
--unit-interval 0.1 \
--alpha 1
