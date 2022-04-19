from decimal import Decimal
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HOME = str(Path.home())
RESULT_PATH = f'{HOME}/da150x/results'

metrics = [
    'acc',
    'acc_val',
    'loss',
    'loss_val',
    'auc',
    'auc_val',
    'f1',
    'f1_val',
]

for metric in metrics:
    filename_csv = f'{RESULT_PATH}/{metric}.csv'
    output_path = f'{RESULT_PATH}/{metric}.png'

    plots = []
    plt.figure(figsize=(8, 8))
    plt.legend(loc='upper left')
    plt.ylabel(metric)
    plt.title(f'a chart with {metric} performance for various configurations')
    plt.xlabel('epoch')

    with open(filename_csv, 'r') as file:
        lines = file.readlines()
        values = {}
        epochs = {}
        last_value = []

        for line in lines[1:]:
            cols = line.split(',')
            network_name = cols[0]
            sample = cols[1]
            batch_size = int(cols[2])
            learning_rate = Decimal(cols[3])
            loss_function = cols[4]
            epoch = int(cols[5])
            value = Decimal(cols[6])

            main_label = f'#N-{network_name}#S-{sample}-#B-{batch_size}-#LR{learning_rate}#LOSS-{loss_function}'
            if main_label not in values:
                values[main_label] = []
            if main_label not in epochs:
                epochs[main_label] = []

            values[main_label].append(value)
            epochs[main_label].append(epoch)

        for label in values:
            last_value.append({'label': label, 'value': values[label][len(values[label]) - 1]})

        last_value.sort(key=lambda x: x['value'], reverse=True)
        for label in last_value:
            label = label['label']
            p, = plt.plot(epochs[label], values[label], label=label)
            plots.append(p)

    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(handles=plots, title='title', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
    plt.tight_layout()
    plt.ylim([min(plt.ylim()) - 0.1 * max(plt.ylim()), max(plt.ylim()) + 0.1 * max(plt.ylim())])
    plt.savefig(output_path)
    plt.close()
