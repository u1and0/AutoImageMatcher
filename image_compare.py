from typing import Iterable, Tuple, List
import numpy as np
from IPython.display import display, Image
import cv2
from skimage.metrics import structural_similarity as compare_ssim


def calculate_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
    """2つの画像のSSIM(Structual Similarity Index)を算出し、
    画像の類似度をfloatで返す。

    Parameters
    ---
    image1, image2: np.ndarray - 画像のバイナリ

    Returns
    ---
    similarity: float - -1~1の範囲で表され、1に近いほど2つの画像がより類似している
    """
    # 画像をグレースケールで読み込む
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 画像をリサイズする（必要に応じて）
    desired_width, desired_height = 300, 300
    gray_image1 = cv2.resize(gray_image1, (desired_width, desired_height))
    gray_image2 = cv2.resize(gray_image2, (desired_width, desired_height))

    # 類似度を計算する（例：構造的類似性指標（SSIM）を使用）
    similarity: float = compare_ssim(gray_image1, gray_image2)
    return similarity


def list_similar_images(tgt: np.ndarray,
                        *images: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    """tgtの画像バイナリに対して類似度の高い順に並べ替えて
    イメージのファイルパスとスコアのlist of tupleで返す

    Test
    ---

    Performance
    ---

    *args形式で渡すとき
    >> %timeit image_compare.list_similar_images(tgt, *image_iter)
    745 ms ± 47.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    listで渡すとき
    %timeit image_compare.list_similar_images(tgt, image_iter)
    705 ms ± 27.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    iterで渡すとき
    %timeit image_compare.list_similar_images(tgt, image_iter)
    648 ns ± 6.1 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

    しかし、データ数が足りなくなっているバグ

    %timeit image_compare.list_similar_images(tgt, *image_iter)
    The slowest run took 7.04 times longer than the fastest. This could mean that an intermediate result is being cached.
    1.67 µs ± 1.76 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

    iterでimages変数に格納して、iterをタプル展開すると速い
    """
    scores: Iterable[float] = (calculate_similarity(tgt, i) for i in images)
    sorted_score = sorted(
        zip(images, scores),  # (画像バイナリ, 類似度スコア)
        key=lambda x: x[1],  # scoreで並べ替え
        reverse=True)  # 降順ソートで[0]がハイスコアになる
    return sorted_score


def find_most_similar_image(tgt: np.ndarray,
                            *images: np.ndarray) -> np.ndarray:
    """
    Parameters
    ---
    tgt: np.ndarray - 比較元の画像, file existならファイルパスとして扱う
    images: Iterable[np.ndarray] - 画像のフルパス

    Returns
    ---
    most_similar_image: np.ndarray - 最も酷似した画像のフルパス

    >>> target_image= cv2.imread("test/145.jpg")
    >>> image_iter = (os.path.join("data", f) for f in os.listdir("data"))
    >>> find_most_similar_image(target_image_path, *image_iter)
    """
    # 降順ソートなので[0]スライスして最もハイスコアの画像を取得
    # (image_path, score)を[0]スライスしてimage_pathだけ取得
    return list_similar_images(tgt, *images)[0][0]


def display_notebook(image_path: str):
    """Jupyter Notebookに指定したパスの画像を表示する"""
    most_similar_image = cv2.imread(image_path)
    _, buf = cv2.imencode(".jpg", most_similar_image)
    display(Image(data=buf.tobytes()))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
