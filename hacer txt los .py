# 1) Colócate en la raíz del proyecto
cd "C:\Users\Vicente Collado\Desktop\Master Thesis\XGBoost\renewable-forecasting-pipeline"

# 2) Vuelca todos los .py REALES (no enlaces) en un único txt
Get-ChildItem -Recurse -Filter *.py -File |
    Where-Object {
        $_.Attributes -notmatch 'ReparsePoint' -and       # quita symlinks
        $_.FullName -notmatch '\\(venv|__pycache__)\\'    # quita carpetas basura
    } |
    ForEach-Object {
        "### $($_.FullName) ###"
        Get-Content -LiteralPath $_.FullName
        ""
    } |
    Set-Content -Path proyecto_codigo.txt -Encoding utf8  # sobrescribe (o crea) el archivo
