import PIL
import tkinter as tk
from PIL import Image, ImageTk
import os
from functools import lru_cache
import threading

PIL.Image.MAX_IMAGE_PIXELS = 11881676800


def get_segment_id_from_string(string):
    return string.split("_")[0]


class FolderImageFrameInfo:
    def __init__(self, folder_image_frame, label_image_frame, img_frame):
        self.folder_image_frame = folder_image_frame
        self.label_image_frame = label_image_frame
        self.img_frame = img_frame


class ImageComparer:
    def __init__(self, folders, inklabel_folder, image_order, max_width=800, max_height=1200, cache_size=10):
        self.folders = folders
        self.inklabel_folder = inklabel_folder
        self.max_width = max_width
        self.max_height = max_height
        self.image_order = image_order
        self.images = self.get_matching_images()
        self.index = 0
        self.cache_size = cache_size

        self.root = tk.Tk()

        self.root.title("Image Comparer")

        self.filename_label = tk.Label(self.root, text="")
        self.filename_label.pack()
        self.filename_label.bind("<Button-1>", self.copy_filename_to_clipboard)

        self.navigation_frame = tk.Frame(self.root)
        self.navigation_frame.pack()

        self.prev_button = tk.Button(self.navigation_frame, text="<<", command=self.prev_image)
        self.prev_button.pack(side="left", expand=True, anchor="e")

        self.next_button = tk.Button(self.navigation_frame, text=">>", command=self.next_image)
        self.next_button.pack(side="right", expand=True, anchor="w")

        self.image_canvas = tk.Canvas(self.root)
        self.image_canvas.pack(side="top", fill=tk.BOTH, expand=True)
        scrollbar_y = tk.Scrollbar(self.root, orient="vertical", command=self.image_canvas.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x = tk.Scrollbar(self.root, orient="horizontal", command=self.image_canvas.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.image_canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        self.image_frames = tk.Frame(self.image_canvas, width=self.max_width * 3, height=self.max_height)
        self.folder_image_frames = []

        self.image_canvas.create_window((0, 0), window=self.image_frames, anchor="nw")

        def on_frame_configure(event):
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))

        # Bind the function to the frame's configure event
        self.image_frames.bind("<Configure>", on_frame_configure)

        for folder in folders:
            folder_image_frame = tk.Frame(self.image_frames)
            folder_label = tk.Label(folder_image_frame, text=os.path.basename(folder))
            folder_label.pack()
            folder_image = tk.Label(folder_image_frame)
            folder_image.pack()
            folder_image_frame.pack(side="left")
            self.folder_image_frames.append(FolderImageFrameInfo(folder_image_frame, folder_label, folder_image))

        self.ink_image_frame = tk.Frame(self.image_frames)
        self.ink_label_label = tk.Label(self.ink_image_frame, text="Ink Labels")
        self.ink_label_label.pack()
        self.ink_image = tk.Label(self.ink_image_frame)
        self.ink_image.pack()
        self.ink_image_frame.pack(side="left")

        self.start_preloading()
        self.update_image_display()

    def start_preloading(self):
        preload_thread = threading.Thread(target=self.load_and_cache_image)
        preload_thread.start()

    def load_and_cache_image(self):
        for i in range(max(0, self.index - 1), min(len(self.images), self.index + 2)):
            for folder in self.folders:
                self.load_image_cached(folder, self.images[i])
            self.load_image_cached(self.inklabel_folder, get_segment_id_from_string(self.images[i]) + "_inklabels.png",
                                   placeholder=True)

    def get_matching_images(self):
        matching_files = list(set.union(*[set(os.listdir(folder)) for folder in self.folders]))
        return sorted(matching_files, key=lambda x: self.image_order.get(get_segment_id_from_string(x),
                                                                         float('-inf')), reverse=True)

    @lru_cache(maxsize=100)
    def load_image_cached(self, folder, filename, placeholder=True):
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            image = Image.open(path)
            image = self.resize_image(image)
            return path, image
        elif placeholder:
            # Create an 'X' placeholder image
            image = Image.new('RGB', (100, 100), color='red')
            return path, image

    def copy_filename_to_clipboard(self, event):
        self.root.clipboard_clear()
        self.root.clipboard_append(get_segment_id_from_string(self.images[self.index]))
        self.root.update()  # now it stays on the clipboard after the window is closed

    def resize_image(self, image):
        width, height = image.size
        ratio = min(self.max_width / width, self.max_height / height)
        new_size = int(width * ratio), int(height * ratio)
        return image.resize(new_size, Image.BILINEAR)

    def update_image_display(self):
        current_image = self.images[self.index]
        self.filename_label.config(text=f"Segment id: {get_segment_id_from_string(current_image)}")

        for i, folder in enumerate(self.folders):
            filename, img = self.load_image_cached(folder, current_image)
            image = ImageTk.PhotoImage(img)
            self.folder_image_frames[i].img_frame.config(image=image)
            self.folder_image_frames[i].img_frame.image = image

        ink_label_image = ImageTk.PhotoImage(self.load_image_cached(self.inklabel_folder, get_segment_id_from_string(
            current_image) + "_inklabels.png", placeholder=True)[1])

        self.ink_image.config(image=ink_label_image)
        self.ink_image.image = ink_label_image

        self.start_preloading()

    def next_image(self):
        self.index = (self.index + 1) % len(self.images)
        self.update_image_display()

    def prev_image(self):
        self.index = (self.index - 1) % len(self.images)
        self.update_image_display()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Usage
    v3 = '../results/greyboxv4_new'
    v4 = '../results/greyboxv4_only_reverse'
    v5_reverse = '../results/greyboxv5_reverse_channel_aug'
    folder3 = '../labels'
    image_order = {'20231005123335': 252872658.0, '20231012184421': 173507978.5, '20231022170900': 155218812.0,
                   '20230702185753': 154629633.5, '20231106155351': 151181172.5, '20231012173610': 142181724.5,
                   '20231005123333':
                       133588200.0, '20231007101615': 122819983.0, '20231031143852': 121577337.5,
                   '20231031143851': 111791347.0,
                   '20230929220924': 99446003.5, '20231106155350': 94503236.0, '20230701020044': 92780785.0,
                   '20231012184420':
                       83153569.5, '20231016151000': 75713336.5, '20231012085431': 60908611.0,
                   '20231024093300': 58613658.5,
                   '20230522181603': 48072778.0, '20230925002745': 41334896.0, '20230925090314': 40987711.5,
                   '20231001164029':
                       34019414.5, '20230827161847': 32744247.0, '20230922174128': 31996743.5,
                   '20230903193206': 28214956.0,
                   '20231004222109': 26136148.0, '20231011111857': 24486950.0, '20231011144857': 23821611.0,
                   '20230904020426':
                       23579711.0, '20230613204956': 23302226.0, '20230926164631': 23049179.5,
                   '20230926164853': 21647513.5,
                   '20230905134255': 21588509.5, '20230909121925': 21149985.0, '20230601192025': 20536616.0,
                   '20230509182749':
                       20004406.5, '20230709155141': 19541563.0, '20230510153006': 19204888.0,
                   '20230901184804': 17781433.5,
                   '20230826211400': 16889650.0, '20230904135535': 16753744.0, '20230919113918': 16730017.5,
                   '20230601193301':
                       16677788.0, '20230625171244': 16638574.0, '20230629215956': 15774059.0,
                   '20230522215721': 14959700.5,
                   '20230620230617': 14252210.5, '20230820203112': 14230582.0, '20230826170124': 14105868.0,
                   '20230620230619':
                       14090848.0, '20230531211425': 13838306.0, '20230531193658': 13086224.0,
                   '20230520175435': 13009565.0,
                   '20230713152725': 12775564.5, '20230627122904': 12537316.5, '20230511215040': 12405763.5,
                   '20230901234823':
                       12338239.5, '20230902141231': 12179544.0, '20230527020406': 11883676.5,
                   '20230627170800': 11628297.5,
                   '20230530172803': 11279499.0, '20230627202005': 11180858.5, '20230516115453': 10409651.5,
                   '20230820174948':
                       10409295.0, '20230701115953': 9967918.0, '20230512123446': 9788895.5,
                   '20230711201157': 9451719.0,
                   '20230526154635': 9385846.5, '20230826135043': 8909695.5, '20230602213452': 8829000.5,
                   '20230605065957':
                       8775821.5, '20230619113941': 8500759.5, '20230606222610': 8456403.5, '20230624144604': 8399360.0,
                   '20230526183725': 8365712.5, '20230611145109': 8355724.5, '20230509173534': 8092302.0,
                   '20230626140105':
                       8092271.0, '20230523002821': 7964972.0, '20230531101257': 7804824.0, '20230530212931': 7715629.0,
                   '20230601201143': 7632988.0, '20230702182347': 7580059.0, '20230624160816': 7518020.0,
                   '20230828154913':
                       7302046.0, '20230619163051': 7182596.5, '20230601204340': 7094981.0, '20230530164535': 7044125.5,
                   '20230603153221': 6980561.5, '20230612195231': 6842359.0, '20230621182552': 6824089.0,
                   '20230819210052':
                       6712093.5, '20230426144221': 6696410.5, '20230608200454': 6668481.0, '20230518130337': 6400431.5,
                   '20230812170020': 6344088.5, '20230522152820': 6257652.0, '20230606105130': 6245804.5,
                   '20230709211458':
                       6209598.0, '20230521113334': 6189495.0, '20230608150300': 6153048.5, '20230504093154': 6107879.5,
                   '20230813_frag_2': 6053496.0, '20230819093803': 5949111.0, '20230820091651': 5831808.0,
                   '20230522182853':
                       5781090.5, '20230519215753': 5437746.0, '20230621122303': 5393038.5, '20230613144727': 5298101.0,
                   '20230604111512': 5271855.0, '20230623160629': 5138164.5, '20230706165709': 5053031.0,
                   '20230801194757':
                       4886183.0, '20230717092556': 4756056.0, '20230604112252': 4749531.5, '20230528112855': 4696392.5,
                   '20230721143008': 4693181.0, '20230624190349': 4678521.5, '20230608222722': 4638521.5,
                   '20230611014200':
                       4638521.5, '20230531121653': 4635687.5, '20230604161948': 4633874.0, '20230517205601': 4551878.0,
                   '20230808163057': 4457237.5, '20230707113838': 4406631.5, '20230705142414': 4246641.5,
                   '20230530025328':
                       4041419.5, '20230515151114': 4013265.5, '20230625194752': 3950837.0, '20230425200944': 3873007.5,
                   '20230505141722': 3770289.5, '20230521114306': 3769835.5, '20230623123730': 3728880.5,
                   '20230529203721':
                       3728756.5, '20230626151618': 3704679.5, '20230505113642': 3656960.5, '20230519195952': 3652998.5,
                   '20230806132553': 3512335.5, '20230518210035': 3501280.0, '20230509160956': 3376977.0,
                   '20230523043449':
                       3337476.0, '20230526015925': 3280348.5, '20230813_real_1': 3230336.0,
                   '20230524173051': 3202969.0,
                   '20230801193640': 3138915.0, '20230525212209': 3134729.5, '20230609123853': 3029703.0,
                   '20230806094533':
                       3017692.0, '20230719103041': 3000123.0, '20230712124010': 2918603.0, '20230422213203': 2898643.5,
                   '20230526175622': 2779751.0, '20230719214603': 2681670.0, '20230521155616': 2587041.5,
                   '20230523191325':
                       2560966.5, '20230525200512': 2550635.5, '20230426114804': 2537813.0, '20230525121901': 2469859.5,
                   '20230424181417': 2447542.0, '20230526002441': 2407370.0, '20230526205020': 2388430.0,
                   '20230516154633':
                       2305804.0, '20230521182226': 2194605.5, '20230515162442': 2112064.0, '20230525190724': 2104768.5,
                   '20230621111336': 2101698.0, '20230518181521': 2099079.0, '20230523233708': 2090625.5,
                   '20230519031042':
                       2083800.0, '20230711222033': 2031978.0, '20230504171956': 1974785.5, '20230504231922': 1900486.0,
                   '20230504151750': 1870881.0, '20230523182629': 1851350.5, '20230524200918': 1834628.5,
                   '20230521193032':
                       1714643.5, '20230712210014': 1669836.5, '20230503120034': 1662914.5, '20230918145743': 1640225.5,
                   '20230918143910': 1628217.0, '20230918024753': 1627328.0, '20230918140728': 1627195.0,
                   '20230918023430':
                       1622627.5, '20230918022237': 1611632.0, '20230918021838': 1598075.0, '20230602092221': 1564238.5,
                   '20230518223227': 1500762.0, '20230526164930': 1479885.5, '20230518191548': 1465994.0,
                   '20230522210033':
                       1379036.5, '20230517214715': 1377383.0, '20230711210222': 1374260.0, '20230521093501': 1340612.0,
                   '20230524005636': 1243710.0, '20230508032834': 1207557.0, '20230424213608': 1207256.5,
                   '20230512112647':
                       1190804.0, '20230421235552': 1146700.5, '20230501042136': 1139619.0, '20230721122533': 1139008.5,
                   '20230520080703': 1088444.5, '20230525234349': 1046814.5, '20230518012543': 1046559.5,
                   '20230425163721':
                       1000584.5, '20230511211540': 997690.0, '20230519140147': 991134.5, '20230513164153': 959083.5,
                   '20230520132429':
                       945516.0, '20230517000306': 942057.5, '20230511094040': 940302.5, '20230505093556': 938597.5,
                   '20230516114341':
                       925847.0, '20230517193901': 914622.5, '20230521104548': 914600.5, '20230524163814': 806040.5,
                   '20230507064642':
                       782553.0, '20230508171353': 778401.0, '20230516112444': 738724.0, '20230525194033': 717009.0,
                   '20230522055405':
                       681643.0, '20230512123540': 677859.0, '20230520191415': 669377.5, '20230525051821': 640381.0,
                   '20230519202000':
                       627765.0, '20230720215300': 618186.0, '20230504225948': 563100.5, '20230512192835': 527279.0,
                   '20230505175240':
                       511051.0, '20230524092434': 486739.5, '20230522151031': 462399.5, '20230511204029': 443238.0,
                   '20230422011040':
                       432979.0, '20230506133355': 421700.0, '20230517180019': 419881.5, '20230519033308': 398153.0,
                   '20230518075340':
                       392645.0, '20230517153958': 383394.0, '20230508181757': 382659.5, '20230519213404': 381433.5,
                   '20230427171131':
                       371797.5, '20230510153843': 363580.0, '20230512211850': 346673.5, '20230517204451': 344752.0,
                   '20230513095916':
                       343119.0, '20230511224701': 325305.5, '20230523034033': 324572.0, '20230504125349': 305170.5,
                   '20230512170431':
                       283679.5, '20230520192625': 267580.0, '20230421192746': 250710.5, '20230525115626': 244668.0,
                   '20230508220213':
                       236812.0, '20230517025833': 235149.5, '20230510170242': 229573.5, '20230506111616': 228929.5,
                   '20230518104908':
                       220106.0, '20230522172834': 209487.5, '20230501040514': 206939.5, '20230519212155': 193494.5,
                   '20230421215232':
                       187840.5, '20230523020515': 182851.0, '20230512094635': 167667.0, '20230512105719': 166997.5,
                   '20230517171727':
                       161654.5, '20230512111225': 149758.5, '20230517104414': 137091.5, '20230505142626': 129104.0,
                   '20230506145829':
                       125879.5, '20230511201612': 121939.5, '20230512120728': 120640.0, '20230505164332': 116720.5,
                   '20230517021606':
                       113614.0, '20230518135715': 103445.5, '20230509163359': 101352.5, '20230505131816': 100129.5,
                   '20230507125513':
                       97431.0, '20230505135219': 95712.0, '20230503213852': 83442.0, '20230504094316': 82609.5,
                   '20230517151648':
                       73177.5, '20230517024455': 72298.0, '20230508080928': 71590.5, '20230513092954': 69709.5,
                   '20230421204550':
                       61537.0, '20230504223647': 59155.5, '20230506142341': 56811.0, '20230514173038': 50085.0,
                   '20230506151750':
                       43158.0, '20230511085916': 43130.0, '20230507175344': 42486.0, '20230506145035': 40613.0,
                   '20230507172452':
                       33549.0, '20230508164013': 29802.0, '20230507175928': 27326.5, '20230517164827': 26787.5,
                   '20230524004853':
                       20793.5, '20230503225234': 16082.5, '20230508131616': 7536.0, '20230506141535': 4687.5,
                   '20230514182829': -1,
                   '20230511150730': -1, '20230505150348': -1, '20230520105602': -1, '20230509144225': -1}
    app = ImageComparer([v3, v4, v5_reverse], folder3, image_order=image_order)
    app.run()
