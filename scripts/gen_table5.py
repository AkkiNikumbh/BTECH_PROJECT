"""Generate table5_statistical_tests.csv and .tex from final_results.csv"""
import pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

TABLES = 'results/tables'
df = pd.read_csv(os.path.join(TABLES, 'final_results.csv'))
print("Columns:", df.columns.tolist())

t5 = df[['Config', 'Wilcoxon_p', 'Cohens_d', 'Significant']].copy()
t5.columns = ['Config', 'Wilcoxon_p', 'Cohens_d', 'Significant']
t5['Wilcoxon_p'] = t5['Wilcoxon_p'].apply(lambda x: f'{float(x):.6f}')
t5['Cohens_d']   = t5['Cohens_d'].apply(lambda x: f'{float(x):.3f}')
t5['Significant'] = t5['Significant'].map({True: 'No', False: 'No', 'True': 'Yes', 'False': 'No'}).fillna('No')

csv_path = os.path.join(TABLES, 'table5_statistical_tests.csv')
t5.to_csv(csv_path, index=False)
print(f'Saved CSV: {csv_path}')

tex_lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Statistical Tests: Explanation vs.\ Blind ($N=10$), Wilcoxon Signed-Rank (one-sided, $\alpha=0.05$)}',
    r'\label{tab:stats}',
    r'\begin{tabular}{lrrl}',
    r'\toprule',
    r'Config & Wilcoxon $p$ & Cohen\textquotesingle s $d$ & Significant \\',
    r'\midrule',
]
for _, row in t5.iterrows():
    sig = 'Yes' if row['Significant'] == 'Yes' else 'No'
    tex_lines.append(f'{row["Config"]} & {row["Wilcoxon_p"]} & {row["Cohens_d"]} & {sig} \\\\')
tex_lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

tex_path = os.path.join(TABLES, 'table5_statistical_tests.tex')
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(tex_lines))
print(f'Saved LaTeX: {tex_path}')
