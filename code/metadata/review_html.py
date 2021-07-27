from pathlib import Path
import json

t1w_path = "/Shared/sinapse/mbrzus/minipig_snapshots/t1w/"
t2w_path = "/Shared/sinapse/mbrzus/minipig_snapshots/t2w/"
out_path = "/home/mbrzus/programming/Cross-Modality-Minipig-Gan/code/metadata/"

with open(f"{out_path}/review_minipig.html", "w") as f:
    f.write("<!DOCTYPE html>")
    f.write("<html>")
    f.write("<head>")
    f.write(
        f'\t<link rel="stylesheet" href="/home/mbrzus/programming/Cross-Modality-Minipig-Gan/code/metadata/styles.css">'
    )
    f.write("</head>")
    f.write("<body>")

    for file in Path(t1w_path).glob("*"):
        t1_path = str(file)
        file_name = t1_path.replace(f"{str(file.parent)}/", "")
        name = file_name[0:28]

        t2_path = t1_path.replace("t1w", "t2w")

        f.write(f"<p>{name}</p>")
        f.write('<div class="row">')

        f.write('\t<div class="column">')
        f.write(f"<p>T1w</p>")
        f.write(f"\t\t<img src={t1_path}>")  # style=\"width:30%\">")
        f.write("\t</div>")

        f.write('\t<div class="column">')
        f.write(f"<p>T2w</p>")
        f.write(f"\t\t<img src={t2_path}>")  # style=\"width:30%\">")
        f.write("\t</div>")

        f.write("</div>")
        f.write("<hr>")

    f.write("</body>")
    f.write("</html>")
