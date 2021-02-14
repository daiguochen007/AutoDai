# 带滑动条的sin绘图
from flexx import flx

class SineExample(flx.Widget):
    def init(self):
        time = [i / 100 for i in range(100)]
        with flx.VBox():
            with flx.HBox():
                # 文本标签
                flx.Label(text='Frequency:')
                # 滑动条设置
                self.slider1 = flx.Slider(min=1, max=10, value=5, flex=1)
                flx.Label(text='Phase:')
                self.slider2 = flx.Slider(min=0, max=6, value=0, flex=1)
            # 绘图控件
            self.plot = flx.PlotWidget(flex=1, xdata=time, xlabel='time',
                                       ylabel='amplitude', title='a sinusoid')

    @flx.reaction
    def __update_amplitude(self, *events):
        global Math
        freq, phase = self.slider1.value, self.slider2.value
        ydata = []
        for x in self.plot.xdata:
            ydata.append(Math.sin(freq * x * 2 * Math.PI + phase))
        self.plot.set_data(self.plot.xdata, ydata)


if __name__ == '__main__':
    m = flx.launch(SineExample)
    flx.run()
