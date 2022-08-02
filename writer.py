def template(fileName):
    start='''
    <!--Author by Yuji-->
    <!DOCTYPE html> 
    <html lang="en">
        <head>
            <title>Summary</title>
            <meta charset="UTF-8">
            <link rel="stylesheet" type="text/css" href="style.css">
            <style type="text/css">
            h1{text-align: center; font-size: 30px}
            p {text-align: center; font-size: 20px; color: #000000}
            h2{text-align: center;}
            h3{text-align: center;}
            .founder{text-align: center;  margin-left: auto; margin-right: auto;}
            </style>
        </head>
        <body>
        <header>
        <h1 style="font-size: 50px" id="header">Summary of %s File</h1>
        </header>
    ''' %fileName

    end='''
        </body>
    </html>
    '''

    return start, end

def write(fileName, val):
    # to open/create a new html file in the write mode
    f = open(fileName + '.html', 'w')

    # the html code which will go in the file GFG.html
    start, end = template(fileName)
    html_template = start + val + end

    # writing the code into the file
    f.write(html_template)

    # close the file
    f.close()



