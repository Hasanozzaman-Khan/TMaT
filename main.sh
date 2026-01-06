# setup the environment
echo Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets_model="gpt-4 gemini gpt-3.5-turbo"

datasets="xsum writing pubmed"

scoring_models="gpt-neo-2.7B"

base_models='PPL PPL_likelihood PPL_logrank PPL_lrr PPL_Fast' 

#evaluate divergence in the black-box setting
echo `date`, Evaluate models in the black-box setting:

for N in $datasets_model; do
    for D in $datasets; do
        for M in $scoring_models; do
            for BM in $base_models; do
                echo `date`, Evaluating on ${D}_${N}.${M1}_${M2} ...
                python ./tda_score.py --reference_model_name ${M} --scoring_model_name ${M} --dataset $D --basemodel $BM \
                                    --dataset_file $data_path/${D}_${N} --output_file $res_path/${D}_${N}.${M}
            done
        done
    done
done