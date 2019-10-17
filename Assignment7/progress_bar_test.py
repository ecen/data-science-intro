#%%
import progressbar
from time import sleep
bar = progressbar.ProgressBar(maxval=100, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i in range(100):
    bar.update(i+1)
    sleep(0.5)
bar.finish()
