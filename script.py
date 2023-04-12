import math

import gradio as gr
import modules.shared as shared
import numpy as np
from transformers import pipeline

temp_range = [0.4, 1.2]
typical_p_range = [0.8, 0.2]
repetition_penalty_range = [1.15, 1.1]
encoder_repetition_penalty_range = [1.05, 1]

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                      top_k=None, device="cpu")

# Parameters which can be customized in settings.json of webui
params = {
    # 'active': True,
    'print_debug': True,
    'temp_lo': 0.4,
    'temp_hi': 1.1,
    'typical_p_lo': 0.8,
    'typical_p_hi': 0.2,
    'repetition_penalty_lo': 1.15,
    'repetition_penalty_hi': 1.1,
    'encoder_repetition_penalty_lo': 1.05,
    'encoder_repetition_penalty_hi': 1,
    'penalty_alpha_lo': 2.5,
    'penalty_alpha_hi': 1
}


def print_debug(text):
    if params['print_debug']:
        print('[Autonomic System Extension] ' + text)


def sigmoid(z):
    a = 1 / (1 + (np.exp((-z))))
    return a


def autonomic_map(dict_list):
    out = []
    mtx = [0, 0, 0, 0, 0, 0]
    print_debug('--------------')
    print_debug('Raw Sentiment Output')
    print_debug('--------------')
    print_debug(str(dict_list))
    print_debug('--------------')
    print_debug('Sentiment Component Contributions')
    print_debug('--------------')
    # Anger
    # anger = sigmoid(15 * mtx[0] - 10) + 0.2 * np.sin(math.pi*mtx[0])**2

    # Anger
    mtx[0] = [d['score'] for d in dict_list if d['label'] == 'anger'][0]
    anger = mtx[0]**3
    out.append(anger)
    print_debug(f'Anger: {np.round(anger, 3)}')

    # Disgust
    mtx[1] = [d['score'] for d in dict_list if d['label'] == 'disgust'][0]
    disgust = mtx[1]
    out.append(disgust)
    print_debug(f'Disgust: {np.round(disgust, 3)}')

    # Fear
    mtx[2] = [d['score'] for d in dict_list if d['label'] == 'fear'][0]
    fear = mtx[2]**2
    out.append(fear)
    print_debug(f'Fear: {np.round(fear, 3)}')

    # Joy
    mtx[3] = [d['score'] for d in dict_list if d['label'] == 'joy'][0]
    joy = 0.75*(mtx[3]+0.1)*np.cos(math.pi*mtx[3])**2
    out.append(joy)
    print_debug(f'Joy: {np.round(joy, 3)}')

    # Neutral is the 4th value

    # Sadness
    mtx[4] = [d['score'] for d in dict_list if d['label'] == 'sadness'][0]
    sadness = 0.75*(mtx[4]+0.1)*np.cos(math.pi*mtx[4])**2
    out.append(sadness)
    print_debug(f'Sadness: {np.round(sadness, 3)}')

    # Surprise
    mtx[5] = [d['score'] for d in dict_list if d['label'] == 'surprise'][0]
    surprise = mtx[5]
    out.append(surprise)
    print_debug(f'Surprise: {np.round(surprise, 3)}')

    # out = anger + disgust + fear + joy + sadness + surprise

    # Normalize if desired
    final = np.sum(out)
    # final = np.sum(out) / np.sum(mtx)

    # Ceiling
    if final > 1:
        final = 1

    print_debug('--------------')
    print_debug(f'Autonomic coefficient: {np.round(final, 2)}')
    print_debug('--------------')

    return final


def range_bias(val, bias):
    out = val[0] + bias * (val[1] - val[0])
    return out


def parameter_map(bias, toggle):
    params_new = [range_bias([params['temp_lo'], params['temp_hi']], bias),
                  range_bias([params['typical_p_lo'], params['typical_p_hi']], bias),
                  range_bias([params['repetition_penalty_lo'], params['repetition_penalty_hi']], bias),
                  range_bias([params['encoder_repetition_penalty_lo'], params['encoder_repetition_penalty_hi']], bias),
                  range_bias([params['penalty_alpha_lo'], params['penalty_alpha_hi']], bias)]

    if toggle == 0:
        fname = "presets/Autonomic_Buffer_A.txt"
    else:
        fname = "presets/Autonomic_Buffer_B.txt"

    with open(fname, "w") as text_file:
        print(f'temperature={params_new[0]}\n', file=text_file)
        print(f'typical_p={params_new[1]}\n', file=text_file)
        print(f'repetition_penalty={params_new[2]}\n', file=text_file)
        print(f'encoder_repetition_penalty={params_new[3]}\n', file=text_file)
        print(f'penalty_alpha={params_new[4]}\n', file=text_file)


def which_params(toggle):
    if toggle == 0:
        buffer = "Autonomic_Buffer_A"
        toggle = 1
    else:
        buffer = "Autonomic_Buffer_B"
        toggle = 0

    return buffer, toggle


def autonomic_update(text, buffer):
    emotions = classifier(text)[0]

    parameter_map(autonomic_map(emotions), buffer)

'''
def activate(x):
    params.update({"active": x})
    if x:
        return [x, 'autonomic_params']
    else:
        return [x, 'LLaMA-Precise']



def preset_modifier(text):
    if params['active']:
        return 'autonomic_params'
    else:
        return text
'''


def ui():
    with gr.Row():
        button_a = gr.Button(value='Autonomic Update', elem_id='load_autonomic')
        # active = gr.Checkbox(value=params['active'], label='Enable autonomic parameter management')
        debug = gr.Checkbox(value=params['print_debug'], label='Print debug information to console')
        buffer_switch = gr.Number(interactive=False, visible=False)
    with gr.Accordion(label='Parameter Ranges', open=False):
        with gr.Row():
            temp_lo = gr.Slider(label='temperature (min arousal)', minimum=0.05, maximum=2, step=0.05, value=params['temp_lo'])
            temp_hi = gr.Slider(label='temperature (max arousal)', minimum=0.05, maximum=2, step=0.05, value=params['temp_hi'])
        with gr.Row():
            typical_p_lo = gr.Slider(label='typical_p (min arousal)', minimum=0, maximum=1, step=0.05, value=params['typical_p_lo'])
            typical_p_hi = gr.Slider(label='typical_p (max arousal)', minimum=0, maximum=1, step=0.05, value=params['typical_p_hi'])
        with gr.Row():
            repetition_penalty_lo = gr.Slider(label='repetition_penalty (min arousal)', minimum=0, maximum=2, step=0.05, value=params['repetition_penalty_lo'])
            repetition_penalty_hi = gr.Slider(label='repetition_penalty (max arousal)', minimum=0, maximum=2, step=0.05, value=params['repetition_penalty_hi'])
        with gr.Row():
            encoder_repetition_penalty_lo = gr.Slider(label='encoder_repetition_penalty (min arousal)', minimum=0, maximum=2, step=0.05, value=params['encoder_repetition_penalty_lo'])
            encoder_repetition_penalty_hi = gr.Slider(label='encoder_repetition_penalty (max arousal)', minimum=0, maximum=2, step=0.05, value=params['encoder_repetition_penalty_hi'])
        with gr.Row():
            penalty_alpha_lo = gr.Slider(label='penalty_alpha (min arousal)', minimum=0, maximum=5, step=0.05, value=params['penalty_alpha_lo'])
            penalty_alpha_hi = gr.Slider(label='penalty_alpha (max arousal)', minimum=0, maximum=5, step=0.05, value=params['penalty_alpha_hi'])

    # active.change(lambda x: activate(x), active, shared.gradio['preset_menu'])
    debug.change(lambda x: params.update({"print_debug": x}), debug, None)
    button_a.click(autonomic_update, [shared.gradio['textbox'], buffer_switch])\
        .then(which_params, buffer_switch, [shared.gradio['preset_menu'], buffer_switch])

    temp_lo.change(lambda x: params.update({"temp_lo": x}), temp_lo, None)
    temp_hi.change(lambda x: params.update({"temp_hi": x}), temp_hi, None)
    typical_p_lo.change(lambda x: params.update({"typical_p_lo": x}), typical_p_lo, None)
    typical_p_hi.change(lambda x: params.update({"typical_p_hi": x}), typical_p_hi, None)
    repetition_penalty_lo.change(lambda x: params.update({"repetition_penalty_lo": x}), repetition_penalty_lo, None)
    repetition_penalty_hi.change(lambda x: params.update({"repetition_penalty_hi": x}), repetition_penalty_hi, None)
    encoder_repetition_penalty_lo.change(lambda x: params.update({"encoder_repetition_penalty_lo": x}), encoder_repetition_penalty_lo, None)
    encoder_repetition_penalty_hi.change(lambda x: params.update({"encoder_repetition_penalty_hi": x}), encoder_repetition_penalty_hi, None)
    penalty_alpha_lo.change(lambda x: params.update({"penalty_alpha_lo": x}), penalty_alpha_lo, None)
    penalty_alpha_hi.change(lambda x: params.update({"penalty_alpha_hi": x}), penalty_alpha_hi, None)
