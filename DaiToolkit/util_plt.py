# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_colormap(num, style="default"):
    """
    get_colormap(num,style="default")
    
    get colormap with given style
    style list: ["PuBuGn","YlGn","BuGn","summer","BrBG","ocean","terrain","Blues","YlGnBu"]
    """
    param_dict = {"PuBuGn": [1., .15],
                  "YlGn": [1., .15],
                  "BuGn": [1., .2],
                  "summer": [1., 0.],
                  "BrBG": [0., 1.],
                  "ocean": [.9, 0.],
                  "terrain": [.9, 0.],
                  "Blues": [1, .15],
                  "YlGnBu": [.1, 1]}
    # set default
    if style == "default":
        style = "BrBG"

    cm = plt.get_cmap(style)(np.linspace(param_dict[style][0], param_dict[style][1], num))
    for i in range(len(cm)):
        if len([x for x in cm[i] if x < 0.95]) == 0:
            cm[i] = cm[i] * 0.9
    return cm


def plt_piechart(temp, title, legend_type={'type': 'right', 'param': 1},
                 font={'family': 'normal', 'weight': 'normal', 'size': 10}, ring=False, style="default"):
    """
    plt_piechart(temp,title,legend_type={'type':'right','param':1},
                  font = {'family':'normal','weight':'normal','size':10},ring=False,style="default")
    
    temp is a data frame with one column/all the pie chart values,index as labels
    use blue green color style, fancy ring chart
    
    legend_type={'type':'bottom',
                 'param':3}  #ncol
                {'type':'right',
                 'param':1}  #right scale
    
    font = {'family':'normal', # ['normal','serif', 'sans-serif', 'fantasy', 'monospace']
            'weight':'normal', # ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']
            'size':10}
    
    """
    matplotlib.rc('font', **font)
    temp = temp / temp.sum().sum()
    # if need self defined color
    #    colors = plt.cm.prism(np.linspace(0.0, 0.1, len(temp)))
    #    colors = plt.cm.Set2(np.arange(len(temp))/float(len(temp)))
    plt.pie(temp.values, startangle=90, colors=get_colormap(len(temp), style))  # ,autopct='%.0f%%',labels=temp.index)

    if ring:
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
    if legend_type["type"] == "right":
        plt.legend(list(map(lambda x, y: str(x) + ": " + str(round(y[0] * 100, 1)) + "%", temp.index, temp.values)),
                   loc='center left', bbox_to_anchor=(legend_type["param"], 0.5))
    if legend_type["type"] == "bottom":
        plt.legend(list(map(lambda x, y: str(x) + ": " + str(round(y[0] * 100, 1)) + "%", temp.index, temp.values)),
                   loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=legend_type["param"])

    plt.axis('equal')
    plt.title(title, fontweight=font["weight"])
    plt.show()


def plt_waterfall2(temp, yunit=None, title="Waterfall Allocation", text_label=True, style="default",
                   xtick_chglne_sptr=None, save_path=None):
    """
    plt_waterfall2(temp,yunit=None,title="Waterfall Allocation",style="default",xtick_chglne_sptr=None)
    
    temp is a pd Series with index(name str) and values(number)
    *sorted in order/allocate from top to bottom
    
    yunit = None
            k
            mm
            bn
            perc
            
    box plot machanism/ deal with postive alloc only
    """
    yunit_dict = {"k": [1000.0, "k"], "mm": [1000000.0, "mm"], "bn": [1000000000.0, "bn"], "perc": [0.01, "%"],
                  None: [1.0, ""]}
    temp = temp.iloc[::-1]
    colors = get_colormap(len(temp.index), style)
    plt.figure(figsize=(9, 5))
    bars = np.transpose([temp.values] * (len(temp.index) + 1))
    plt.bar(list(range(len(temp.index) + 1)), bars[0], color=[colors[0]] + ['white'] * (len(temp.index) - 1) + [colors[0]],
            edgecolor='white', width=1)
    for i in range(1, len(temp.index)):
        plt.bar(list(range(len(temp.index) + 1)), bars[i], bottom=bars[:i].sum(axis=0),
                color=[colors[i]] + ['white'] * (len(temp.index) - 1 - i) + [colors[i]] + ['white'] * i,
                edgecolor="white", width=1)
    if text_label:
        plt.text(0, temp.sum(), str(round(temp.sum() / yunit_dict[yunit][0], 1)) + yunit_dict[yunit][1], ha='center',
                 va='bottom', weight="bold")
        for i in range(len(temp.index)):
            plt.text(i + 1, temp.sum() - temp[::-1][:i].sum(),
                     str(round(temp[::-1].iloc[i] / yunit_dict[yunit][0], 1)) + yunit_dict[yunit][1], ha='center',
                     va='bottom', weight="bold")
    xlabels = ["All"] + list(temp.index)[::-1]
    if xtick_chglne_sptr is not None:
        xlabels = ["\n".join(x.split(xtick_chglne_sptr)) for x in xlabels]
    locs, labels = plt.yticks()
    plt.yticks(locs, [str(int(float(x) / yunit_dict[yunit][0])) + yunit_dict[yunit][1] for x in locs])
    plt.xticks(list(range(len(temp.index) + 1)), xlabels, fontsize=11)
    plt.title(title, fontweight='bold')
    # plt.grid(ls="--",alpha=0.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def plt_waterfall(temp, yunit=None, title="Waterfall Allocation", text_label=True, style="default",
                  xtick_chglne_sptr=None, save_path=None):
    """
    plt_waterfall(temp,yunit=None,title="Waterfall Allocation",style="default",xtick_chglne_sptr=None)
    
    temp is a pd Series with index(name str) and values(number)
    *sorted in order/allocate from top to bottom
    
    yunit = None
            k
            mm
            bn
            perc
            
    bar plot machanism/ deal with postive/neg alloc
    """
    yunit_dict = {"k": [1000.0, "k"], "mm": [1000000.0, "mm"], "bn": [1000000000.0, "bn"], "perc": [0.01, "%"],
                  None: [1.0, ""]}
    temp_bottom = temp.sum() - temp.cumsum()
    # temp_top = temp_bottom + temp
    colors = get_colormap(len(temp.index), style)
    plt.figure(figsize=(9, 5))
    for i in range(len(temp)):
        if i == 0:
            plt.bar(0, pd.Series(temp[::-1].iloc[i], index=["All"]), width=1, bottom=pd.Series([0], index=["All"]),
                    color=colors[i], edgecolor='white')
        else:
            plt.bar(0, pd.Series(temp[::-1].iloc[i], index=["All"]), width=1,
                    bottom=pd.Series([temp[::-1].cumsum().iloc[i - 1]], index=["All"]), color=colors[i],
                    edgecolor='white')
    plt.bar(list(range(len(temp.index) + 1)), pd.Series([0], index=["All"]).append(temp), width=1,
            bottom=pd.Series([0], index=["All"]).append(temp_bottom),
            color=np.append([colors[0]], colors[::-1], axis=0), edgecolor='white')
    if text_label:
        plt.text(0, temp.sum(), str(round(temp.sum() / yunit_dict[yunit][0], 1)) + yunit_dict[yunit][1], ha='center',
                 va='bottom', weight="bold")
        for i in range(len(temp.index)):
            if temp.iloc[i] >= 0:
                plt.text(i + 1, (temp.sum() - temp.cumsum() + temp).iloc[i],
                         str(round(temp.iloc[i] / yunit_dict[yunit][0], 1)) + yunit_dict[yunit][1], ha='center',
                         va='bottom', weight="bold")
            else:
                plt.text(i + 1, (temp.sum() - temp.cumsum()).iloc[i],
                         str(round(temp.iloc[i] / yunit_dict[yunit][0], 1)) + yunit_dict[yunit][1], ha='center',
                         va='bottom', weight="bold")
    xlabels = ["All"] + list(temp.index)
    if xtick_chglne_sptr is not None:
        xlabels = ["\n".join(x.split(xtick_chglne_sptr)) for x in xlabels]
    locs, labels = plt.yticks()
    plt.yticks(locs, [str(int(float(x) / yunit_dict[yunit][0])) + yunit_dict[yunit][1] for x in locs])
    plt.xticks(list(range(len(temp.index) + 1)), xlabels, fontsize=11)
    plt.title(title, fontweight='bold')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def plt_lineplot(temp, yunit=None, title="Line Plot", figsize=(9, 5), linewidth=3, marker=None,
                 legend_type={'type': 'right', 'param': 1},
                 style="default", save_path=None):
    """
    plt_lineplot(temp,yunit=None,title="Line Plot",text_label=True,style="default",xtick_chglne_sptr=None,save_path=None)
    
    temp is a pd dataframe with index and columns
    marker = o/None/*/^
    yunit = None
            k
            mm
            bn
            perc
    """
    yunit_dict = {"k": [1000.0, "k"], "mm": [1000000.0, "mm"], "bn": [1000000000.0, "bn"], "perc": [0.01, "%"],
                  None: [1.0, ""]}
    colors = get_colormap(len(temp.columns), style)
    plt.figure(figsize=figsize)
    for i, c in enumerate(temp.columns):
        plt.plot(temp[c], label=c, linewidth=linewidth, marker=marker, color=colors[i])
    locs, labels = plt.yticks()
    plt.yticks(locs, [str(int(float(x) / yunit_dict[yunit][0])) + yunit_dict[yunit][1] for x in locs])
    plt.title(title, fontweight='bold')
    plt.grid(ls="--", alpha=0.5)
    if legend_type["type"] == "right":
        plt.legend(loc='center left', bbox_to_anchor=(legend_type["param"], 0.5))
    if legend_type["type"] == "bottom":
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=legend_type["param"])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


########################################### test
if __name__ == "__main__":
    df = pd.DataFrame(list(range(8)), index=list(range(8)), columns=["0"])
    for style in ["PuBuGn", "YlGn", "BuGn", "summer", "BrBG", "ocean", "terrain"]:
        # plt_piechart(df,style,{'type':'right','param':0.9},style=style)
        plt_waterfall(df["0"], yunit=None, title="Waterfall Allocation", style=style)
