# 上傳到 GitHub 的步驟

## 1. 在 GitHub 上創建新倉庫

1. 登入 GitHub (https://github.com)
2. 點擊右上角的 "+" → "New repository"
3. 填寫倉庫資訊：
   - Repository name: `Drafter` (或您喜歡的名稱)
   - Description: `AI 自動繪圖系統 - 3D 到 2D 工程圖轉換工具`
   - 選擇 Public 或 Private
   - **不要**勾選 "Initialize this repository with a README"（我們已經有 README.md）
4. 點擊 "Create repository"

## 2. 連接本地倉庫到 GitHub

在終端機中執行以下命令（將 `YOUR_USERNAME` 替換為您的 GitHub 用戶名）：

```bash
# 添加遠端倉庫
git remote add origin https://github.com/YOUR_USERNAME/Drafter.git

# 或者使用 SSH（如果您已設置 SSH key）
# git remote add origin git@github.com:YOUR_USERNAME/Drafter.git

# 查看遠端倉庫
git remote -v
```

## 3. 上傳到 GitHub

```bash
# 推送主分支到 GitHub
git push -u origin master

# 或者如果您的分支名稱是 main
# git branch -M main
# git push -u origin main
```

## 4. 驗證

上傳完成後，在瀏覽器中打開：
`https://github.com/YOUR_USERNAME/Drafter`

您應該能看到所有檔案和 README.md 的內容。

## 後續更新

當您修改程式碼後，使用以下命令更新 GitHub：

```bash
# 查看變更
git status

# 添加變更的檔案
git add .

# 提交變更
git commit -m "描述您的變更"

# 推送到 GitHub
git push
```

## 注意事項

- `.gitignore` 已經設置好，會自動忽略：
  - `output/` 目錄（生成的檔案）
  - `__pycache__/`（Python 快取）
  - 大型檔案（.stp, .dxf, .dwg 等）
- 如果需要上傳測試檔案，可以手動添加到 Git
