from PyQt5.Qt import QApplication
from diffrend.model import load_obj
from diffrend.gl.GLRenderer import create_OGL_window


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Renderer.\n Usage: ' + sys.argv[0] +
                                                 '--obj object-filename' +
                                                 '--renderer gl21')
    parser.add_argument('--obj', type=str, help='Wavefront OBJ file path', default='../data/bunny.obj')
    parser.add_argument('--renderer', type=str, help='Renderer to use', default='gl21')

    args = parser.parse_args()

    app = QApplication(sys.argv)

    obj = load_obj(args.obj)

    if args.renderer == 'gl21':
        window = create_OGL_window(2, 1, obj=obj, title='OpenGL 2.1 Renderer')
    elif args.renderer == 'gl':
        window = create_OGL_window(None, None, obj=obj, title='OpenGL Renderer')
    else:
        raise ValueError('Renderer {} not implemented.'.format(args.rederer))

    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()

