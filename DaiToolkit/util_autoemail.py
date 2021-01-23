from jinja2 import PackageLoader, Environment
from win32com import client as win32


class AutoEmail(object):
    def __init__(self, subject=None, _html=None, _to_address=None, _cc_address=None, create_draft=True,
                 template_name=None, template_data=None, attachment_path=None):
        self.subject = subject
        self._html = _html
        self._to_address = _to_address
        self._cc_address = _cc_address
        self.create_draft = create_draft
        self.template_name = template_name
        self.template_data = template_data
        self.attachment_path = attachment_path

    @property
    def to_address(self):
        if isinstance(self._to_address, str):
            self._to_address = [self._to_address]
        return ''.join([email + ';' for email in self._to_address])

    @property
    def cc_address(self):
        if self._cc_address is None:
            return self._cc_address
        if isinstance(self._cc_address, str):
            self._cc_address = [self._cc_address]
        return ''.join([email + ';' for email in self._cc_address])

    @property
    def html(self):
        if self._html is None:
            self.html = self.create_html_content()
        return self._html

    def create_html_content(self):
        if self.template_data is None:
            raise ValueError('Need template data')
        file_loader = PackageLoader('templates', '')
        env = Environment(loader=file_loader)
        template = env.get_template(self.template_name)
        html_content = template.render(data=self.template_data)
        return html_content

    def create_outlook_item(self):
        outlook = win32.Dispatch('outlook.application')
        mail = outlook.CreateItem(0)
        mail.To = self.to_address
        mail.Subject = self.subject
        mail.HtmlBody = self.html

        if self.cc_address is not None:
            mail.CC = self.cc_address
        if self.attachment_path is not None:
            if isinstance(self.attachment_path, list):
                for att in self.attachment_path:
                    mail.Attachments.Add(Source=att)
            else:
                mail.Attachments.Add(self.attachment_path)
        return mail

    def send_email(self):
        mail = self.create_outlook_item()
        if self.create_draft:
            mail.Display(False)
        else:
            mail.Send()
