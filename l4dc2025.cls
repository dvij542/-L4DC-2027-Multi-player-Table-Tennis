\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{l4dc2025}[2023/08/22 v1.0 L4DC 2025 Proceedings]

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[pmlr]{jmlr}

\jmlrvolume{vol 283}
\jmlryear{2025}
\jmlrproceedings{PMLR}{Proceedings of Machine Learning Research}

\iffinalsubmission
  \editors{N. Ozay, L. Balzano, D. Panagou, A. Abate}
  \jmlrworkshop{7th Annual Conference on Learning for Dynamics and Control}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Temporary workaround fixing issues with \includegraphics (concerns LaTeX2e
%% releases after October 2019) [jsieber; 2020-11-12]
\def\set@curr@file#1{\def\@curr@file{#1}}

%% Workaround for Overleaf [Dean Huang, Will Redman, 2023-10-25]
\makeatletter
\let\Ginclude@graphics\@org@Ginclude@graphics
\makeatother
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\coltauthor}[1]{\author{#1}}