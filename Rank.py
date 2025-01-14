import os
import csv
import sqlite3
from tkinter import Tk, Label, Entry, filedialog
from PIL import Image, ImageTk


class ImageRankingTool:
    def __init__(self, db_path, output_file):
        self.db_path = db_path
        self.output_file = output_file
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        
        # Fetch all images and metadata from the database
        self.cursor.execute("SELECT id, filename, followers, following, numPosts FROM ImageTable")
        self.images_data = self.cursor.fetchall()
        self.current_index = 0
        self.rankings = []
        
        # Initialize GUI
        self.root = Tk()
        self.root.title("Image Ranking Tool")
        self.image_label = Label(self.root)
        self.image_label.pack()
        self.entry = Entry(self.root)
        self.entry.pack()
        
        # Bind the Enter key to save the rank
        self.entry.bind("<Return>", lambda event: self.save_rank())

        self.display_image()
        self.root.mainloop()

    def display_image(self):
        if self.current_index < len(self.images_data):
            _, filename, follow_count, following_count, num_posts = self.images_data[self.current_index]
            img_path = os.path.join(filename)
            try:
                img = Image.open('Cropped/' + img_path)
                img.thumbnail((950, 950))  # Resize for display
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk
            except FileNotFoundError:
                print(f"Error: File {img_path} not found.")
                self.current_index += 1
                self.display_image()
        else:
            self.finish()

    def save_rank(self):
        rank = self.entry.get()
        if rank.isdigit() and 1 <= int(rank) <= 100:
            rank = int(rank)
            row_data = list(self.images_data[self.current_index])  # Get current image row
            row_data.append(rank)  # Add ranking to row data
            self.rankings.append(row_data)  # Add to rankings
            self.current_index += 1
            self.entry.delete(0, 'end')
            self.display_image()
        else:
            print("Please enter a valid rank between 1 and 100.")

    def finish(self):
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "filename", "followCount", "followingCount", "numPosts", "ranking"])
            writer.writerows(self.rankings)
        print(f"Ranking complete! Results saved to {self.output_file}.")
        self.connection.close()
        self.root.quit()


def main():
    db_path = filedialog.askopenfilename(
        title="Select SQLite Database", filetypes=[("SQLite Database", "*.db")]
    )
    if db_path:
        output_file = filedialog.asksaveasfilename(
            title="Save Rankings As", defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if output_file:
            ImageRankingTool(db_path, output_file)


if __name__ == "__main__":
    main()
