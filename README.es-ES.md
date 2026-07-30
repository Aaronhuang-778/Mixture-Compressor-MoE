

<p align="center" width="10%">
<img src="imgs/logo.png" style="width: 30%" align=center> 
</p>

# Compresor de Mezcla para LLMs de Mezcla de Expertos: Más Ventajas

[Wei Huang](https://aaron-weihuang.com/), [Yue Liao](https://scholar.google.com/citations?user=mIt-3fEAAAAJ&hl=en), [Jianhui Liu](https://scholar.google.com/citations?user=n1JW-jYAAAAJ&hl=en), [Ruifei He](https://scholar.google.com/citations?user=P7IL0hkAAAAJ), [Haoru Tan](), [Shiming Zhang*](), [Hongsheng Li](https://scholar.google.com/citations?user=BN2Ze-QAAAAJ&hl=zh-CN), [Si Liu*](https://scholar.google.com/citations?user=-QtVtNEAAAAJ&hl=en) y [Xiaojuan Qi*](https://scholar.google.com/citations?user=bGn0uacAAAAJ&hl=en) (* autor correspondiente)

[![arXiv](https://img.shields.io/badge/MCMoE-2410.06270-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06270)



![WX20241009-191322@2x](imgs/WX20241009-191322@2x.png)

Compresión extrema de Modelos de Lenguaje Grandes de Mezcla de Expertos. La versión actual incluye soporte para:

- MC-MoE para cuantización precisa solo de pesos (Pesos=**1.5～2.5 bits**).
- MC-MoE para poda dinámica en línea eficiente (relación de compresión adicional **> 10%**)
- Soporte para MoE-LLM preentrenados (actualmente solo se admite **Mixtral 8 $\times$ 7b** y **Mixtral 8 $\times$ 22b**)
- El proceso de despliegue y decuantización real se basa en `HQQ` y `GPTQ` para cuantización por grupos estándar. La cuantización estática de MC-MoE también puede transferirse a otros tipos de técnicas, como la cuantización vectorial (libro de códigos).

## Instalación
```sh
conda create -n mcmoe python=3.10 -y
conda activate mcmoe
git clone https://github.com/Aaronhuang-778/MC-MoE
cd MC-MoE
pip install --upgrade pip 
pip install -r requirements.txt
```

Para la cuantización real y el despliegue del modelo comprimido, utilizamos [HQQ](https://github.com/mobiusml/hqq) para decuantizar los LLMs MoE con 1/2/3 bits. Hemos modificado el proceso de almacenamiento y decuantización de los pesos de 1 bit en HQQ. 

Asegúrese de tener una versión de PyTorch 2 que coincida con su versión de CUDA: https://pytorch.org/.

## Catálogo de Precisiones de Expertos
Proporcionamos el ancho de bits resuelto para cada experto de **Mixtral 8 $\times$ 7b** en `./experts_mixture_bit_selection/`. Puede utilizar directamente nuestros resultados proporcionados o generarlos usted mismo.

## Uso

**Obtenga rápidamente el modelo comprimido con `./scripts/quant.sh`**. Ya hemos proporcionado todos los resultados intermedios necesarios de **Mixtral 8 $\times$ 7b** en este código.

```sh
# Replace the path with yours, for example:
Model_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1"
Saving_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1-2.5b"
Precision_Path="./experts_mixture_bit_selection/experts_mixture_bitwidth_combination_20bit.pkl"
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path} --pack --save --saving_path ${Saving_Path}

```

Inferencia eficiente con **Cuantización de Precisión Mixta Precargada** y **Poda Dinámica en Línea**: Este ejemplo muestra la demostración del modelo Mixtral-8x7B de 2.5 bits, el consumo total de memoria GPU estática es de alrededor de **16 GB** y la memoria en ejecución es de alrededor de **19 GB** .

```python
import os
import torch
from transformers import AutoTokenizer
from inference import load_quantized_model
from expert_weight import  replace_with_dynamic_rank
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.is_available()


kwargs = {"device_map": 'auto',
          "torch_dtype": "torch.float16"}
######## Input your save_dir of quantized model########
save_dir = "/mnt/models/mistralai/Mixtral-8x7B-v0.1-2.5bit"
model = load_quantized_model(save_dir, kwargs)

######### Choose if you want to use dynamic pruning or not ##########
args = None
model = replace_with_dynamic_rank(model, args, block_range=10)
######### Choose if you want to use dynamic pruning or not ##########

tokenizer = AutoTokenizer.from_pretrained(save_dir)
prompt = "You are a writer. Please write a short story about two llamas in a forest"
prompt_template=f'''{prompt}
'''

inputs = tokenizer(prompt_template, return_tensors="pt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
inputs.input_ids = inputs.input_ids.to(device)

inputs.attention_mask = inputs.attention_mask.to(device)
# Generate
outputs = model.generate(inputs.input_ids, 
                        max_new_tokens=512,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        )

print(tokenizer.decode(outputs[0]))
```



**Proceso detallado de MC-MoE**. 

1. Primero, necesitamos generar los factores de expertos para la asignación de ancho de bits en cada bloque MoE:

   Descargue la primera parte de los datos de entrenamiento C4 `c4-train.00000-of-01024.json` desde [allenai/c4](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz). Por favor, guárdelo en `./data` y organice los conjuntos de datos de la siguiente manera: 
```
./data
|-- build.py
|-- c4-train.00000-of-01024.json
|-- dataset.py
|-- math_calib_construction.py
`-- math_pretrain_style.json
```
Ejecute `./scripts/factors.sh` para generar las `frecuencias de activación`, `pesos de activación` y `pérdida de cuantización` de cada experto.

```sh
# Please run this script in ./scripts/factors.sh

# Your local model path
Model_Path=""
python awareness.py ${Model_Path} --calibration c4

# for example:
Model_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1"
python awareness.py ${Model_Path} --calibration c4

```
También puede cambiar los datos de `--calibration` a `math` para realizar la calibración para conocimiento y tarea específicos. Ya hemos proporcionado el archivo de factores del conjunto de datos `c4`, por favor verifique:

```
|-- experts_act_frequency.pkl
|-- experts_act_weight.pkl
|-- experts_quant_loss.pkl
```

2. En segundo lugar, utilizamos los factores para resolver el problema de asignación de ancho de bits a través de `precision_solver.py` (genere los archivos de factores anteriores antes de ejecutar el código de asignación de ancho de bits):

```sh
python precision_solver.py
```

Este solver guardará las configuraciones óptimas de expertos en cada bloque MoE:

```sh
./experts_mixture_bit_selection
|-- experts_mixture_bitwidth_combination_12bit.pkl
|-- experts_mixture_bitwidth_combination_13bit.pkl
|-- experts_mixture_bitwidth_combination_14bit.pkl
|-- experts_mixture_bitwidth_combination_15bit.pkl
|-- experts_mixture_bitwidth_combination_16bit.pkl
|-- experts_mixture_bitwidth_combination_17bit.pkl
|-- experts_mixture_bitwidth_combination_18bit.pkl
|-- experts_mixture_bitwidth_combination_19bit.pkl
|-- experts_mixture_bitwidth_combination_20bit.pkl

# 12 bit means the total bit-width of 8 experts in one MoE block, the average bit-width of experts is 12/8=1.5bit.
```

3. Tercero, podemos ejecutar el código de cuantización para comprimir los parámetros estáticos de los LLMs MoE:

Ejecute `./scripts/quant.sh` para cuantizar los LLMs MoE y probar los resultados de perplexidad.

```sh
# Your local model path
Model_Path=""
# Expected experts precisions file
Precision_Path=""
##### fake quantization to test the performance of MC-MoE #####
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path}

# for example:
Model_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1"
Precision_Path="./experts_mixture_bit_selection/experts_mixture_bitwidth_combination_16bit.pkl"
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path}

```

agregue `--pach` y `--save` para guardar el modelo cuantizado real:

```sh
# Your local model path
Model_Path=""
# Your expected saving path
Saving_Path=""
# Expected experts precisions file
Precision_Path=""

##### real quantization and model pack for compact storage #####
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path} --pack --save --saving_path ${Saving_Path}

# for example:
Model_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1"
Saving_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1-2.05bit"
Precision_Path="./experts_mixture_bit_selection/experts_mixture_bitwidth_combination_16bit.pkl"
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path} --pack --save --saving_path ${Saving_Path}

```

4. Finalmente, puede cargar los LLMs MoE cuantizados y habilitar la inferencia con poda dinámica con nuestra demostración de inferencia proporcionada:

```sh
python inference_demo.py
```



donde también puede elegir si habilitar la poda dinámica para mejorar la eficiencia de inferencia (línea `17~18`):

```python
from expert_weight import  replace_with_dynamic_rank
args = None
model = replace_with_dynamic_rank(model, args, block_range=10)
```
Demostración detallada de inferencia:

```python
import os
import torch
from transformers import AutoTokenizer
from inference import load_quantized_model
from expert_weight import  replace_with_dynamic_rank
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.is_available()


kwargs = {"device_map": 'auto',
          "torch_dtype": "torch.float16"}
######## Input your save_dir of quantized model########
save_dir = "/mnt/models/mistralai/Mixtral-8x7B-v0.1-2.5bit"
model = load_quantized_model(save_dir, kwargs)

######### Choose if you want to use dynamic pruning or not ##########
args = None
model = replace_with_dynamic_rank(model, args, block_range=10)
######### Choose if you want to use dynamic pruning or not ##########

tokenizer = AutoTokenizer.from_pretrained(save_dir)
prompt = "You are a writer. Please write a short story about two llamas in a forest"
prompt_template=f'''{prompt}
'''

inputs = tokenizer(prompt_template, return_tensors="pt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
inputs.input_ids = inputs.input_ids.to(device)

inputs.attention_mask = inputs.attention_mask.to(device)
# Generate
outputs = model.generate(inputs.input_ids, 
                        max_new_tokens=512,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        )

print(tokenizer.decode(outputs[0]))
```



## Evaluación

![m2](imgs/m2.png)

Utilizamos el marco de trabajo [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/2a47159caff00135b026f724ace2a2011f3c7621) (commit 2a47159) para evaluar el rendimiento de los LLMs MoE con cuantización simulada. El comando que utilizamos para la evaluación con LM Harness es el siguiente:

```sh
# accelerate launch \
#     --num_processes=1 \
#     --ipex \
#     -m lm_eval --model hf \
#     --model_args pretrained=model_path,dtype=float16,parallelize=True \
#     --tasks piqa,boolq,arc_challenge,arc_easy,hellaswag,winogrande,mmlu,mathqa \
#     --batch_size 32 \
```



## Proyectos Relacionados

[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://github.com/OpenGVLab/OmniQuant)

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models](https://github.com/Aaronhuang-778/SliM-LLM)

[Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models](https://github.com/Lucky-Lance/Expert_Sparsity)

[Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark](https://github.com/UNITES-Lab/moe-quantization)

[Half-Quadratic Quantization (HQQ)](https://github.com/mobiusml/hqq)
