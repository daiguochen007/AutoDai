# -*- coding: utf-8 -*-
###########################################################
#
#    This is a toolkit file for excel
#
###########################################################
import os
import re

import win32com.client as win32
from PIL import ImageGrab
from PyPDF2 import PdfFileWriter, PdfFileReader
from pywintypes import com_error


def excel_colnum_str(n):
    """map column number to excel column letter"""
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string


def rgb_to_hex(rgb):
    """
    get color for excel from rgb
    
    rgb_to_hex((255,255,255) #white
    rgb_to_hex((118,238,198)) #blue green
    
    excel green/good:
    Interior.Color = rgb_to_hex((198,239,206))
    Font.Color = rgb_to_hex((0,97,0))
    
    excel yellow/warning:
    Interior.Color = rgb_to_hex((255,235,156))
    Font.Color = rgb_to_hex((156,101,0))
    
    excel red/bad:
    Interior.Color = rgb_to_hex((206,198,255)) # red
    Font.Color = rgb_to_hex((6,0,156))
    """
    rgb = (rgb[2], rgb[1], rgb[0])
    """for setting rgb color in excel"""
    strValue = '%02x%02x%02x' % rgb
    iValue = int(strValue, 16)
    return iValue


def excel_color(color_type):
    """
    get color for excel from rgb
    
    Good/Bad/Netural/White/Black
    """
    if color_type == "Good":
        return {"Interior": rgb_to_hex((198, 239, 206)), "Font": rgb_to_hex((0, 97, 0))}
    elif color_type == "Bad":
        return {"Interior": rgb_to_hex((206, 198, 255)), "Font": rgb_to_hex((6, 0, 156))}
    elif color_type == "Neutral":
        return {"Interior": rgb_to_hex((255, 235, 156)), "Font": rgb_to_hex((156, 101, 0))}
    elif color_type == "White":
        return rgb_to_hex((255, 255, 255))
    elif color_type == "Black":
        return rgb_to_hex((0, 0, 0))
    elif color_type == "Green":
        return rgb_to_hex((0, 160, 0))


# basic excel table format
def excel_tbl_bscfmt(worksheet, df, index=False, startcol=0, startrow=0):
    """
    excel_tbl_bscfmt(worksheet,df,index=False,startcol=0,startrow=0)
    
    start col/row follows pandas to_excel(from 0)
    """
    if len(df) > 0:
        scol = excel_colnum_str(startcol + 1)
        srow = str(startrow + 1)
        if index:
            endcol = excel_colnum_str(startcol + len(df.columns) + 1)
        else:
            endcol = excel_colnum_str(startcol + len(df.columns))
        worksheet.Range(scol + srow + ":" + endcol + srow).Interior.Color = 1
        worksheet.Range(scol + srow + ":" + endcol + srow).Font.Color = rgb_to_hex((255, 255, 255))
        worksheet.Range(scol + srow + ":" + endcol + srow).Font.Bold = True
        worksheet.Range(scol + srow + ":" + endcol + srow).HorizontalAlignment = 3
        worksheet.Range(scol + srow + ":" + endcol + srow).VerticalAlignment = 2
        for i in range(1, 5):
            worksheet.Range(scol + srow + ":" + endcol + str(startrow + len(df) + 1)).Borders(i).LineStyle = 1


def excel_rangeborder(worksheet, str_range, border_type="All Borders", thickness=1):
    """
    set border for range

    border_type:
        "All Borders"
        "Outside Borders"
        "Thick Box Border"
    """
    if border_type == "All Borders":
        for i in range(1, 5):
            worksheet.Range(str_range).Borders(i).LineStyle = 1
            worksheet.Range(str_range).Borders(i).Weight = thickness

    if border_type == "Thick Box Border":
        sc, sr, ec, er = [y for x in str_range.split(":") for y in re.split('(\d+)', x) if y != ""]
        worksheet.Range(sc + sr + ":" + ec + sr).Borders(3).LineStyle = 1
        worksheet.Range(sc + sr + ":" + ec + sr).Borders(3).Weight = 3
        worksheet.Range(sc + er + ":" + ec + er).Borders(4).LineStyle = 1
        worksheet.Range(sc + er + ":" + ec + er).Borders(4).Weight = 3
        worksheet.Range(sc + sr + ":" + sc + er).Borders(1).LineStyle = 1
        worksheet.Range(sc + sr + ":" + sc + er).Borders(1).Weight = 3
        worksheet.Range(ec + sr + ":" + ec + er).Borders(2).LineStyle = 1
        worksheet.Range(ec + sr + ":" + ec + er).Borders(2).Weight = 3

    if border_type == "Outside Borders":
        sc, sr, ec, er = [y for x in str_range.split(":") for y in re.split('(\d+)', x) if y != ""]
        worksheet.Range(sc + sr + ":" + ec + sr).Borders(3).LineStyle = 1
        worksheet.Range(sc + sr + ":" + ec + sr).Borders(3).Weight = thickness
        worksheet.Range(sc + er + ":" + ec + er).Borders(4).LineStyle = 1
        worksheet.Range(sc + er + ":" + ec + er).Borders(4).Weight = thickness
        worksheet.Range(sc + sr + ":" + sc + er).Borders(1).LineStyle = 1
        worksheet.Range(sc + sr + ":" + sc + er).Borders(1).Weight = thickness
        worksheet.Range(ec + sr + ":" + ec + er).Borders(2).LineStyle = 1
        worksheet.Range(ec + sr + ":" + ec + er).Borders(2).Weight = thickness


def excel_setprintarea(worksheet, print_area, orientation="P", margins=0):
    """
    set print area
    fit as one page and set print area in excel
    
    worksheet is win32com worksheet
    print_area = "$A$1:$I$90"
    orientation="P" for Portrait / "L" for Landscape
    """
    worksheet.PageSetup.Zoom = False
    worksheet.PageSetup.FitToPagesTall = 1
    worksheet.PageSetup.FitToPagesWide = 1
    worksheet.PageSetup.PrintArea = print_area

    worksheet.PageSetup.TopMargin = margins
    worksheet.PageSetup.LeftMargin = margins
    worksheet.PageSetup.RightMargin = margins
    worksheet.PageSetup.BottomMargin = margins
    worksheet.PageSetup.HeaderMargin = 0
    worksheet.PageSetup.FooterMargin = 0

    worksheet.PageSetup.CenterVertically = True
    worksheet.PageSetup.CenterHorizontally = True
    if orientation == "P":
        worksheet.PageSetup.Orientation = 1
    elif orientation == "L":
        worksheet.PageSetup.Orientation = 2
    else:
        pass


def xlsxwriter_dftoExcel(df, worksheet, index=True, header=True, startcol=0, startrow=0):
    """
    xlsxwriter writing df to excel
    function for writing df to excel with xlsx
    when trying not to use df.to_excel
    
    worksheet = xlsxwriter.workbook.add_worksheet('Page 1')
    """
    sc = startcol + 1
    sr = startrow + 1
    # index if True
    if index:
        for i, x in enumerate(df.index):
            worksheet.write(excel_colnum_str(sc) + str(sr + i + 1), df.index[i])
        sc += 1
    # header if True
    if header:
        for i, x in enumerate(df.columns):
            if df.columns[i] == df.columns[i]:
                worksheet.write(excel_colnum_str(sc + i) + str(sr), df.columns[i])
        sr += 1
        # 2 coordinitors
    for x, df_rows in enumerate(df.values):
        for y, v in enumerate(df_rows):
            if v == v:
                worksheet.write(excel_colnum_str(sc + y) + str(sr + x), v)


def excel_to_pdf(excel_path, worksheet_list=[]):
    dir_path = os.path.dirname(excel_path)
    excel_name = os.path.basename(excel_path)
    pdf_name = ".".join(excel_name.split('.')[:-1]) + ".pdf"
    pdf_path = dir_path.replace('/', '\\') + '\\' + pdf_name
    pdf_path_pages = [dir_path.replace('/', '\\') + '\\' + '.'.join(excel_name.split('.')[:-1]) + '_' + x + '.pdf' for x
                      in worksheet_list]

    excel = win32.DispatchEx('Excel.Application')
    excel.visible = False
    wb = excel.Workbooks.Open(excel_path)
    for w, p in zip(worksheet_list, pdf_path_pages):
        wb.Worksheets([w]).Select()
        wb.ActiveSheet.ExportAsFixedFormat(0, p)
    wb.Close()

    def append_pdf(finput, foutput):
        [foutput.addPage(finput.getPage(page_num)) for page_num in range(finput.numPages)]

    with open(pdf_path, 'wb') as of:
        output = PdfFileWriter()
        nf_list = []
        for n in pdf_path_pages:
            nf = open(n, 'rb')
            append_pdf(PdfFileReader(nf), output)
            nf_list.append(nf)
            print("[" + n + "] finished!")
        output.write(of)

        for nf in nf_list:
            nf.close()

        for n in pdf_path_pages:
            os.remove(n)


class ExcelFile(object):
    @classmethod
    def open(cls, filename):
        obj = cls()
        obj._open(filename)
        return obj

    def __init__(self):
        self.app = None
        self.workbook = None

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()
        return False

    def _open(self, filename):
        excel_pathname = os.path.abspath(filename)
        if not os.path.exists(excel_pathname):
            raise IOError('No such excel file: %s', filename)
        try:
            self.app = win32.DispatchEx('Excel.Application')
            self.app.Visible = 0
        except:
            raise OSError('Fail to start Excel')
        try:
            self.workbook = self.app.Workbooks.Open(excel_pathname, ReadOnly=True)
        except:
            self.close()
            raise IOError('Fail to open %s' % filename)

    def close(self):
        if self.workbook is not None:
            self.workbook.Close(SaveChanges=False)
            self.workbook = None
        if self.app is not None:
            self.app.Visible = 0
            self.app.Quit()
            self.app = None


def excel_to_img(fn_excel, fn_image, page=None, _range=None):
    output_ext = os.path.splitext(fn_image)[1].upper()
    if output_ext not in ('.GIF', '.PNG', '.BMP'):
        print('Unsupported format %s' % output_ext)

    if _range is not None and page is not None and '!' not in _range:
        _range = "%s!%s" % (page, _range)
    with ExcelFile.open(fn_excel) as excel:
        if _range is None:
            if page is None:
                page = 1
            try:
                rng = excel.workbook.Sheets(page).UsedRange
            except com_error:
                raise Exception("Fail locating used cell range on page %s" % page)
        else:
            try:
                rng = excel.workbook.Application.Range(_range)
            except com_error:
                raise Exception("Fail loading range %s" % (_range))

        xlScreen = 1
        xlBitmap = 2
        retries, success = 100, False
        while not success:
            try:
                rng.CopyPicture(xlScreen, xlBitmap)
                im = ImageGrab.grabclipboard()
                im.save(fn_image, fn_image[-3:])
                success = True
            except (com_error, AttributeError) as e:
                print(e)
                retries -= 1
                if retries == 0: raise Exception('retried 100 times fail')
