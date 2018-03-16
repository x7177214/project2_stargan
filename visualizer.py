import numpy as np
from collections import OrderedDict
display_port=8097
display_id=1
display_winsize=256
name="face_exp"
import torch
#
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].data.cpu().float().numpy()
    def log_restore(img):
       return  (np.exp(img * np.log(256) / 255) - 1)
    if image_numpy.shape[0] == 1:
        image_numpy = (np.exp(np.tile(image_numpy, (3, 1, 1))*np.log(256)/255)-1)/255
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0,0,1) * 255.0
    return image_numpy.astype(imtype)

class Visualizer():
    def __init__(self):
        self.display_id = display_id
        self.win_size = display_winsize
        self.name = name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=display_port)

    def display_current_results(self, visual_dict):
        for key, value in visual_dict.items():
            visual_dict[key]=tensor2im(value)

        visuals=OrderedDict(sorted(visual_dict.items(), key=lambda x: x, reverse=True))


        if self.display_id > 0:  # show images in the browser
            ncols = 0
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

    def plot_current_errors(self, epoch, counter_ratio, error):

        errors = OrderedDict(sorted(error.items(), key=lambda x: x[1], reverse=True))

        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)



