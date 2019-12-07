import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


###
# score calculation
###
class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns evaluation results.
            - precision
            - recall
            - dice score
        """
        # prevent from nan computation
        epsilon = 0.00001
        hist = self.confusion_matrix

        # precision, recall(sensitivity), and dice(f1-score)
        precision = np.diag(hist) / (hist.sum(axis=0) + epsilon)
        recall = np.diag(hist) / (hist.sum(axis=1) + epsilon)
        dice = (2 * precision * recall) / (precision + recall + epsilon)

        return precision[1], recall[1], dice[1]
               
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

###
# display prediction and groundtruth masks
###
def blend(image, mask, alpha=0.4):
    return image * (1.0 - alpha) + mask * alpha

def showImage(image, lblmask, predmask, outname):
    lblmask = encodeColor(mask=lblmask, maxNum=1)
    predmask = encodeColor(mask=predmask, maxNum=1)
    blend_lbl = blend(image=image, mask=lblmask).astype(np.uint8)
    blend_pred = blend(image=image, mask=predmask).astype(np.uint8)

    image = Image.fromarray(image).rotate(90)
    blend_lbl = Image.fromarray(blend_lbl).rotate(90)
    blend_pred = Image.fromarray(blend_pred).rotate(90)
    
    fig = plt.figure()
    ax = plt.subplot(131)
    ax.set_title('Original Image')
    plt.imshow(image)
    ax = plt.subplot(132)
    ax.set_title('Groundtruth')
    plt.imshow(blend_lbl)
    ax = plt.subplot(133)
    ax.set_title('Prediction')
    plt.imshow(blend_pred)

    fig.savefig(outname, dpi=300, bbox_inches='tight')
    plt.close()

def get_pascal_labels():
    return np.asarray([[0, 0, 0], [255,0,0], [0,255,0], 
                       [0,0,255], [255,255,255]])

def encodeColor(mask, maxNum):
    label_colours = get_pascal_labels()
    r = mask.copy()
    g = mask.copy()
    b = mask.copy()
    for ll in range(0, maxNum+1):
        r[mask == ll] = label_colours[ll, 0]
        g[mask == ll] = label_colours[ll, 1]
        b[mask == ll] = label_colours[ll, 2]

    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb