import triplets_generator,sqlite3
def create():
    conn=sqlite3.connect('representations.db')
    cur = conn.cursor()
    # try:
    #    cur.execute("DROP TABLE IMAGES")
    # except sqlite3.OperationalError:
    #    pass
    cur.execute("CREATE TABLE IF NOT EXISTS IMAGES (id INTEGER PRIMARY KEY, path VARCHAR(255) , matrix BLOB)")
    try:
        cur.execute("CREATE UNIQUE INDEX idx_images_path ON images(path)")
    except:
        pass

    conn.commit()

def update_db(model,dataset="data_train5/"):
    conn = sqlite3.connect('representations.db')
    cur = conn.cursor()
    classes = triplets_generator.get_subdirectories(dataset)
    print ("Updating representations in db")

    for Id, c in enumerate(classes):
        batch=[]
        batch_files=[]
        pos_dir = os.path.join(dataset, c)
        imgs_pos = os.listdir(pos_dir)

        for idx in range(0, len(imgs_pos)):
            img = triplets_generator.img_to_np(os.path.join(dataset,c + '/' + imgs_pos[idx]))
            batch.append(img)
            batch_files.append(imgs_pos[idx])

        print ("Calculating representations for %s"%c)
        preds =model.predict(np.asarray(batch))
        for name, matrix in zip(batch_files, preds):
            name = "%s/%s" % (c, name)
            cur.execute("REPLACE INTO images VALUES (?,?,?)", (None, name, json.dumps(matrix.tolist())))
        conn.commit()

