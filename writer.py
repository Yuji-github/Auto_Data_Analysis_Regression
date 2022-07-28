# import module
import codecs

def write(name, val):
    # to open/create a new html file in the write mode
    f = open(name + '.html', 'w')
    # the html code which will go in the file GFG.html
    html_template = val

    # writing the code into the file
    f.write(html_template)

    # close the file
    f.close()



