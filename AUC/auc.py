

'''
假设有一个训练好的二分类器对10个正负样本（正例5个，负例5个）预测，得分按高到低排序得到的最好预测结果为[1, 1, 1, 0, 1, 0, 1, 0, 0, 0]。
描点方式按照样本预测结果的得分高低从左至右开始遍历。从原点开始，每遇到1便向y轴正方向移动y轴最小步长1个单位，这里是1/5=0.2；
每遇到0则向x轴正方向移动x轴最小步长1个单位，这里也是0.2。
'''
def get_roc(pos_prob,y_true):
    pos = y_true[y_true==1]
    neg = y_true[y_true==0]
    y = y_true[pos_prob.argsort()[::-1]]
    tpr_all = [0] ; fpr_all = [0]
    tpr = 0 ; fpr = 0
    x_step = 1/float(len(neg))
    y_step = 1/float(len(pos))
    area=0    # 用于计算AUC
    for i in range(len(pos_prob)):
        if y[i] == 1:
            tpr += y_step
        else:
            area += tpr*x_step

    return area


'''
假设总共有（m+n）个样本，其中正样本m个，负样本n个，总共有mn个样本对，计数，正样本预测为正样本的概率值大于负样本预测为正样本的概率值记为1，
累加计数，然后除以（mn）就是AUC的值。
'''


def get_roc(pos_prob, y_true):
    pos = y_true[y_true == 1]
    neg = y_true[y_true == 0]
    y = y_true[pos_prob.argsort()[::-1]]
    tpr_all = [0];
    fpr_all = [0]

    tpr = 0;
    fpr = 0

    x_step = 1 / float(len(neg))

    y_step = 1 / float(len(pos))

    y_sum = 0
    for i in range(len(pos_prob)):

        if y[i] == 1:

            tpr += y_step

            tpr_all.append(tpr)

            fpr_all.append(fpr)

        else:

            fpr += x_step

            fpr_all.append(fpr)

            tpr_all.append(tpr)

            y_sum += tpr

    return tpr_all, fpr_all, y_sum * x_step



'''
排序复杂度：O(log2(P + N))(P为正样本数，N为负样本数)

计算AUC复杂度：O(P + N)
'''



