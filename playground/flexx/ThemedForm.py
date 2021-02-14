#主题表单
from flexx import flx


class ThemedForm(flx.Widget):
    CSS = """
    .flx-Button {
        background: #7c76d1;
    }
    .flx-LineEdit {
        border: 2px solid #65cbcc;
    }
    """

    def init(self):
        with flx.HFix():
            with flx.FormLayout() as self.form:
                self.b1 = flx.LineEdit(title='Name:', text='Hola')
                self.b2 = flx.LineEdit(title='Age:', text='Hello world')
                self.b3 = flx.LineEdit(title='Favorite color:', text='Foo bar')
                flx.Button(text='Submit1')
                # flx.Widget(flex=1)  #有间隙空行

            with flx.FormLayout() as self.form:
                self.b4 = flx.LineEdit(title='Name:', text='Hola')
                self.b5 = flx.LineEdit(title='Age:', text='Hello world')
                self.b6 = flx.LineEdit(title='Favorite color:', text='Foo bar')
                flx.Button(text='Submit2')
                flx.Widget(flex=1)  # 没有间隙空行的


if __name__ == '__main__':
    m = flx.launch(ThemedForm)
    flx.run()
