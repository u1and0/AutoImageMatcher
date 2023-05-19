import os
from typing import Iterable, Tuple, List
from more_itertools import sort_together
from IPython.display import display, Image
import cv2
from skimage.metrics import structural_similarity as compare_ssim


def calculate_similarity(image1: str, image2: str) -> float:
    """2つの画像のSSIM(Structual Similarity Index)を算出し、
    画像の類似度をfloatで返す。

    Parameters
    ---
    image1, image2: str - 画像のパス

    Returns
    ---
    similarity: float - -1~1の範囲で表され、1に近いほど2つの画像がより類似している
    """
    # 画像をグレースケールで読み込む
    gray_image1 = cv2.cvtColor(cv2.imread(image1), cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(cv2.imread(image2), cv2.COLOR_BGR2GRAY)

    # 画像をリサイズする（必要に応じて）
    desired_width, desired_height = 300, 300
    gray_image1 = cv2.resize(gray_image1, (desired_width, desired_height))
    gray_image2 = cv2.resize(gray_image2, (desired_width, desired_height))

    # 類似度を計算する（例：構造的類似性指標（SSIM）を使用）
    similarity: float = compare_ssim(gray_image1, gray_image2)
    return similarity


def list_similar_images(tgt: str, *images: str) -> List[Tuple[str, float]]:
    """tgtの画像パスに対して類似度の高い順に並べ替えて
    イメージのファイルパスとスコアのlist of tupleで返す

    Test
    ---
    >>> target_image_path = "test/145.jpg"
    >>> image_iter = (os.path.join("data", f) for f in os.listdir("data"))
    >>> list_similar_images(target_image_path, *image_iter)
    [('data/main.png', 0.5650305022604248), ('data/102274a.jpg', 0.5154026862664451), ('data/airplane-aircraft-top.jpg', 0.456384255911045), ('data/safe_image-2.jpeg', 0.4073903438613619), ('data/logi-mart_51194z220-50.jpeg', 0.4057643478428405), ('data/34_22_8_030_c.jpg', 0.3974721837163576), ('data/charm_SP19-128433D.jpg', 0.3873456006834926), ('data/id_files_24637.jpg', 0.3783472176316682), ('data/JiDU-ROBO-1-Baidu-First-Car-Business.jpeg', 0.37578363321646746), ('data/ships_01_01.jpg', 0.3746375666922977), ('data/img_6040b1cdb6545cc1b2ac3f36c5a23f14170115.jpg', 0.3708120481084022), ('data/img_1.jpg', 0.3640941363848973), ('data/large_200401_ddg_01.jpg', 0.347653122094046), ('data/61707494.jpg', 0.3432303259721921), ('data/3054514a.jpg', 0.3412307936593937), ('data/images01.jpg', 0.31516459534971897), ('data/mainimg_c0021_01.jpg', 0.30934171052318993), ('data/title-1651123563536.jpeg', 0.29885418872658237), ('data/JS_Kirisame(DD-104).jpg', 0.2589797730852747), ('data/20171212_01_01.jpg', 0.2477250926633316), ('data/big_main10015782_20200806113200000000.jpg', 0.2309059971267855), ('data/20191212_workcar_021-650x433.jpg', 0.17055055741480313), ('data/file.jpeg', 0.14732205261130585), ('data/20230509-00000065-asahi-000-2-view.jpg', 0.0765127005141863)]

    Performance
    ---

    *args形式で渡すとき
    >> %timeit image_compare.list_similar_images(target_image, *image_iter)
    745 ms ± 47.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    listで渡すとき
    %timeit image_compare.list_similar_images(target_image, image_iter)
    705 ms ± 27.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    iterで渡すとき
    %timeit image_compare.list_similar_images(target_image, image_iter)
    648 ns ± 6.1 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

    しかし、データ数が足りなくなっているバグ

    %timeit image_compare.list_similar_images(target_image, *image_iter)
    The slowest run took 7.04 times longer than the fastest. This could mean that an intermediate result is being cached.
    1.67 µs ± 1.76 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

    iterでimages変数に格納して、iterをタプル展開すると速い
    """
    scores: Iterable[float] = (calculate_similarity(tgt, i) for i in images)
    sorted_score = sorted(
        zip(images, scores),  # (画像パス, 類似度スコア)
        key=lambda x: x[1],  # scoreで並べ替え
        reverse=True)  # 降順ソートで[0]がハイスコアになる
    return sorted_score


def find_most_similar_image(target_image_path: str, *images: str) -> str:
    """
    Parameters
    ---
    target_image_path: str - 比較元の画像
    images: Iterable[str] - 画像のフルパス

    Returns
    ---
    most_similar_image: str - 最も酷似した画像のフルパス
    """
    # 降順ソートなので[0]スライスして最もハイスコアの画像を取得
    # (image_path, score)を[0]スライスしてimage_pathだけ取得
    return list_similar_images(target_image_path, *images)[0][0]


def display_notebook(image_path: str):
    """Jupyter Notebookに指定したパスの画像を表示する"""
    most_similar_image = cv2.imread(image_path)
    _, buf = cv2.imencode(".jpg", most_similar_image)
    display(Image(data=buf.tobytes()))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
