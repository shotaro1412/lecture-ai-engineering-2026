# 第2回 LoRA実行結果

2026年11月2日に、利用者のGoogleアカウントのColabをCodexが操作して実行した結果です。講師のreferenceデータの転記ではありません。提出者本人が手動でセルを操作したとは記載していません。

## ファイル

- `baseline/`：元の架空講座FAQで実行した演習の結果。出力は実行済みノートブックから抽出し、講座の採点関数で再集計して表示値との一致を確認。
- `library/`：独自の架空図書室FAQで実行した宿題の結果とデータ。
- `result.json`：モデル・設定・学習前後の指標・退行・知識質問の結果。
- `eval_before.json` / `eval_after.json`：21問それぞれの回答原文と採点。
- `train.jsonl` / `valid.jsonl` / `test.jsonl` / `gold.json`：独自FAQの分割済みデータと正解。

## 再実行

`notebooks/day02_library_lora.ipynb` をColabで開いて自分のDriveへコピーし、T4 GPUを選択して上から実行してください。モデルのrevision、ライブラリの固定版、生成設定は元の演習と同じです。Google AI StudioのAPIキーは使いません。

学習と評価は同じFAQを入力に含めます。質問文は分割間で重複しませんが、未知のFAQ文書に対する一般化を示す実験ではありません。学習後の基盤モデル変数にもadapterが付くため、比較をやり直すときは新しいランタイムで最初から実行してください。

## 実行者の記録

`executed_training_myself=false` は、提出者本人の手動実行と区別するためです。実学習は行っており、`training_executed_in_this_run=true` と `training_executed_by` に記録しています。元の演習の既定フラグを正規化した理由はbaseline/result.jsonのprovenance_noteに残しています。

元のノートブックの知識質問表は、FAQへの意味的な根拠の有無を質問文の `q in gold` で代用しているため、対照質問の表示がFalseになる場合があります。回答原文は保持し、answer.mdではFAQ本文を読んで解釈しています。`asserted` はフラグがfalseという分類名で、根拠ありの対照質問なら正しい応答です。
