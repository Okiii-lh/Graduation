# coding=utf-8
"""
@File    :   input_name.py    
@Contact :   13132515202@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2020/3/7 20:38   LiuHe      1.0         None
"""
from tkinter import *
from tkinter import messagebox

root = Tk()
root.title("输入姓名")
root.geometry('300x100')

l1 = Label(root, text="姓名：")
l1.pack()
xls_text = StringVar()
xls = Entry(root, textvariable=xls_text)
xls_text.set(" ")
xls.pack()


def on_click():
    x = xls_text.get()
    string = str("xls名：%s " %(x))
    messagebox.showinfo(title='aaa', message=string)
    root.quit()
    root.destroy()


Button(root, text="确认", command = on_click).pack()
root.mainloop()