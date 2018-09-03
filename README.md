# Partial-Person-ReID
******************************************************************************************************************

**The source code**: Spatial Feature Reconstruction with Pyramid Pooling for Partial Person Re-identification 
CVPR18: [Deep Spatial Feature Reconstruction for Partial Person Re-identification: Alignment-free Approach](http://openaccess.thecvf.com/content_cvpr_2018/papers/He_Deep_Spatial_Feature_CVPR_2018_paper.pdf), Arxiv18 

The project provides the training and testing code for partial person re-id, using [Pytorch](https://pytorch.org/)

## Instllation
*****************************************************************************************************************
It's recommended that you create and enter a python virtual environment, if versions of the packages required here conflict with yours.

Other packages are specified in `requirements.txt`

## Daset Preparation
Inspired by Houjing Huang's [person-reid-triplet-loss-baseline](https://github.com/huanghoujing/person-reid-triplet-loss-baseline) project, you can follow his guidance.

## Experiment Setting:
1. Backbone: ResNet-50, `stride = 1` in the last conv block.
2. Input image size: `384 & times 192`

## Person Re-identification
### Result on Market1501
   \begin{table*}[t]
  \centering
  \small
  %\fontsize{6.5}{8}\selectfont
  \caption{Performance comparison on Market1501 and CHUK03. R1: rank-1. mAP: mean Accuracy Precision.}
  \label{tab2}
    \begin{tabular}{|l|l|l|c|c|c|c|c|c|c|c|}
    \hline
    \multicolumn{2}{|l|}{\multirow{3}{*}{Method}} &
    \multicolumn{4}{c|}{Market1501}&\multicolumn{4}{c|}{CHUK03} \cr \cline{3-10}
    \multicolumn{2}{|c|}{~}&\multicolumn{2}{c|}{single query}&\multicolumn{2}{c|}{multiple query}&\multicolumn{2}{c|}{Labeled}&\multicolumn{2}{c|}{Detected}\cr\cline{3-10}
     \multicolumn{2}{|c|}{~}& R1 &mAP &R1 &mAP &R1 &mAP &R1 &mAP  \cr \hline
     \multirow{6}{*}{Part-based}  &Spindle (CVPR17) \cite{zhao2017spindle}&76.50&-&-&-&-&-&-&-\cr
    &MSCAN (CVPR17) \cite{li2017learning}&80.31&57.53&86.79&66.70&-&-&-&- \cr
    &DLPAP (CVPR17) \cite{zhao2017deeply}&81.00&63.40&-&-&-&-&-&-\cr
    &AlignedReID (Arxiv17) \cite{zhang2017alignedreid}&91.80&79.30&-&-&-&-&-&-\cr
    &PCB (Arxiv17) \cite{sun2017beyond}&92.30&77.40&-&-&-&-&61.30&57.50\cr \hline
     \multirow{3}{*}{Mask-guided}  &SPReID (CVPR18) \cite{kalayeh2018human}& 92.54&\bf 81.34&-&-&-&\-&-&-\cr
    &MGCAM (CVPR18) \cite{song2018mask}&83.79 &74.33&-&-&50.14 &50.21&46.71&46.87 \cr
    &MaskReID (Arxiv18) \cite{qi2018maskreid} & 90.02 &75.30 &93.32 &82.29& -&- &- &- \cr \hline
    \multirow{5}{*}{Pose-guided}  &PDC (ICCV17) \cite{su2017pose}&84.14&63.41&-&-&-&-&-&- \cr
    &PABR (Arxiv18) \cite{suh2018part}&90.20&76.00&93.20&82.70&-&-&-&-\cr
    &Pose-transfer (CVPR18) \cite{liu2018pose}&87.65&68.92&-&-&33.80&30.50&30.10&28.20 \cr
    &PN-GAN (Arxiv17) \cite{qian2017pose}&89.43&72.58&-&-&-&-&-&-\cr
    &PSE (CVPR18) \cite{sarfraz2017pose}&87.70&69.00&-&-&-&-&27.30&30.20\cr \hline
    \multirow{3}{*}{Attention-based}  &DuATM (CVPR18) \cite{si2018dual}&91.42&76.62&-&-&-&-&-&- \cr
    &HA-CNN (CVPR18) \cite{li2018harmonious}&91.20&75.70&93.80&82.80&44.40&41.00&41.70&38.60\cr
    &AACN (CVPR18) \cite{xu2018attention}&85.90&66.87&89.78&75.10&-&-&-&- \cr \hline
   \multicolumn{2}{|l|}{Baseline (ResNet-50+tri)}&88.18&73.85&92.25&80.96&-&-&60.43&54.24 \cr
   \multicolumn{2}{|l|}{DSR (CVPR18) \cite{he2018deep}}&91.26&75.62&93.45&82.44&-&-&61.78&56.87 \cr
   \multicolumn{2}{|l|}{SFR (ours)}&\color{red}\bf 93.04&\color{red}  81.02&\color{red}94.84& \color{red} 85.47&\color{red}\bf 67.29 &\color{red} \bf 61.47&\color{red}\bf 63.86& \color{red} \bf 58.97\cr\hline
    \end{tabular}
    \vspace{1.2em}
\end{table*}
