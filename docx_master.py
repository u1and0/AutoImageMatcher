from zipfile import ZipFile
from docx import Document


def extract_images_from_docx(docx_path: str) -> list[tuple[str, bin]]:
    """docxファイルをzipファイルとして開いて
    word/mediaから始まる主に画像ファイルを取得し、
    ( 画像ファイル名, 画像バイナリ)のタプルを複数含んだリストとして取得する。
    """
    # python-docxを使う方法
    #
    # doc = Document(docx_path)
    # images = []
    #
    # for rel in doc.part.rels.values():
    #     if "image" in rel.reltype:
    #         image_path = rel.target_part.partname[1:]
    #         image_data = rel.target_part.blob
    #         images.append((image_path, image_data))
    #
    # return images
    picture_list = []
    with ZipFile(docx_path, "r") as doxz:
        for item in doxz.infolist():
            if item.filename.startswith("word/media"):
                name = item.filename
                buf = doxz.read(item.filename)
                picture_list.append((name, buf))
    return picture_list


def image_iter(docx_file: str):
    # docxファイルから画像を抽出する
    image_list = extract_images_from_docx(docx_file)

    # 画像の情報を表示する
    for image_path, image_data in image_list:
        yield image_path, image_data  # 一部のデータのみ表示


if __name__ == "__main__":
    for p, d in image_iter("sample.docx"):
        print(p, d[:20])

# result
# word/media/image1.jpeg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00'
# word/media/image2.jpeg b'\xff\xd8\xff\xdb\x00\x84\x00\x05\x03\x04\x04\x04\x03\x05\x04\x04\x04\x05\x05\x05'
# word/media/image3.jpeg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
# word/media/image4.jpeg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00'
# word/media/image5.jpeg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00'
# word/media/image6.jpeg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x01^\x01^\x00\x00'
# word/media/image7.jpeg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x02\x00\x1c\x00\x1c\x00\x00'
# word/media/image8.jpeg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'

# Example of Usage replace_image_in_docx()
# replace_image_in_docx("sample.docx", "word/media/image1.png", "new_image.png")