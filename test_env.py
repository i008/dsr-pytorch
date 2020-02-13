try:
    import nbformat
    from nbconvert import PythonExporter
    import glob 
except Exception as e:
    print('\033[91m' + "TEST FAILED")
    print('PLEASE INSTALL REQUIRED PACKAGES')
    print(e)
    
def load_source(nb_file):
    with open(nb_file) as nb:
        nbsource = nbformat.reads(nb.read(), nbformat.NO_CONVERT)
        source, _ = exporter.from_notebook_node(nbsource)
        return source

def filter_imports(source):
    return [l for l in source.split('\n') if l.startswith('import')]

if __name__ == '__main__':
    exporter = PythonExporter()
    all_imports = []
    for notebook in glob.glob('*.ipynb'):
        print("Gathering imports ... ", notebook)
        all_imports.extend(filter_imports(load_source(notebook)))
        
    test_imports = '\n'.join(all_imports)
    try:
        print("Testing all imports ...")
        exec(test_imports)
        print('\033[92m' + "     ALL GOOD    ")
    except Exception as e:
        print('\033[91m' + "    TEST FAILED    ")
        print(e)
