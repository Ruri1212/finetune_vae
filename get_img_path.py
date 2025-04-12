import os
import argparse

def gather_image_paths(root_dir, valid_extensions=('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
    """
    指定したディレクトリ以下の全画像ファイルのパスをリストとして返す。

    :param root_dir: 探索するルートディレクトリ
    :param valid_extensions: 画像ファイルとみなす拡張子のタプル（小文字で記述）
    :return: 画像ファイルパスのリスト
    """
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(valid_extensions):
                full_path = os.path.join(dirpath, filename)
                image_paths.append(full_path)
    return image_paths

def write_paths_to_file(paths, output_file):
    """
    画像パスのリストをテキストファイルに1行ずつ書き出す。

    :param paths: 画像パスのリスト
    :param output_file: 出力先のテキストファイルパス
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in paths:
            f.write(f"{path}\n")

def main():
    parser = argparse.ArgumentParser(
        description="指定ディレクトリ以下の全画像ファイルのパスをtxtファイルに書き出します。"
    )
    parser.add_argument(
        "--directory", type=str, required=True,
        help="画像ファイルを探索するルートディレクトリのパス"
    )
    parser.add_argument(
        "--output", type=str, default="image_paths.txt",
        help="出力するテキストファイルのパス（デフォルト: image_paths.txt）"
    )
    args = parser.parse_args()

    paths = gather_image_paths(args.directory)
    write_paths_to_file(paths, args.output)
    print(f"総計 {len(paths)} 個の画像パスを {args.output} に書き出しました。")

if __name__ == "__main__":
    main()
