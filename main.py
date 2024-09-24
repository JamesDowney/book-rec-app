from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from lxml import html
import webbrowser
import requests
import recommendation_engine
import matplotlib.pyplot
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib
matplotlib.use('agg')

# This relied heavily on information from https://docs.python.org/3/library/tkinter.html

root = tk.Tk()
img = tk.PhotoImage(file='book-64.png')
root.iconphoto(False, img)
root.title("Book Recommendations")
root.geometry("800x960")
root.grid_columnconfigure((0, 1), weight=1, minsize=100)
book_var = tk.StringVar()


def start_progress_bar():
    book_progress_bar['mode'] = "indeterminate"
    book_progress_bar.start(10)


def stop_progress_bar():
    book_progress_bar.stop()
    book_progress_bar['mode'] = "determinate"
    book_progress_bar.step(0)


def submit_with_loading():
    start_progress_bar()
    threading.Thread(target=submit).start()


def submit():
    global recommended_books, recommended_features, queried_book_features
    global book_titles_with_query, book_titles_without_query, queried_book_title
    submission = book_var.get()
    if len(submission) == 0:
        stop_progress_bar()
        messagebox.showerror("Error", "Submission field must not be blank.")
        return
    for x in book_suggestions.get_children():
        book_suggestions.delete(x)

    try:
        recommended_books, recommended_features, queried_book_features, queried_book_title = recommendation_engine.recommend_books(
            submission,
            15,
            category_weight=category_weight.get(),
            numerical_weight=numerical_weight.get(),
            description_weight=description_weight.get()
        )
    except Exception as e:
        stop_progress_bar()
        messagebox.showerror(
            "Error", "Submission not found, please try a different book title.")
        return

    book_var.set(queried_book_title)

    for x in range(len(recommended_books)):
        book_suggestions.insert("", 'end', iid=x, values=(
            recommended_books.iloc[x].get('Title'),
            recommended_books.iloc[x].get('Author'),
            recommended_books.iloc[x].get("Score"),
            recommended_books.iloc[x].get('Categories')
        ), tags=('even_row' if x % 2 == 0 else 'odd_row'))

    book_titles_with_query = recommended_books['Title'].tolist()
    book_titles_with_query.insert(0, queried_book_title)

    book_titles_without_query = recommended_books['Title'].tolist()

    stop_progress_bar()
    root.after(0, update_visualization)


def display_heatmap(recommended_features, queried_book_features, book_titles, submission):
    combined_features = np.vstack(
        [queried_book_features, recommended_features])

    truncated_titles = [title if len(
        title) <= 20 else title[:17] + '...' for title in book_titles]

    heatmap_figure, heatmap_axes = matplotlib.pyplot.subplots(figsize=(8, 6))

    sns.heatmap(
        cosine_similarity(combined_features),
        xticklabels=truncated_titles,
        yticklabels=truncated_titles,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.7},
        linewidths=0.5,
        ax=heatmap_axes
    )
    heatmap_axes.set_xticklabels(
        truncated_titles, rotation=30, ha='right', fontsize=8)
    heatmap_axes.set_yticklabels(truncated_titles, fontsize=8)

    heatmap_axes.set_title(f'{submission} Heatmap', fontsize=12)

    matplotlib.pyplot.tight_layout()

    return heatmap_figure


def display_bar_chart(recommended_features, queried_book_features, book_titles, submission):
    similarities = cosine_similarity(
        queried_book_features, recommended_features).flatten()

    bar_figure, bar_axes = matplotlib.pyplot.subplots(figsize=(8, 6))
    bar_axes.bar(book_titles, similarities, color='skyblue')
    bar_axes.set_xlabel('Book Titles')
    bar_axes.set_ylabel('Similarity Score')
    bar_axes.set_title('Similarity Scores for Recommended Books')
    bar_axes.set_xticks(range(len(book_titles)))
    bar_axes.set_xticklabels(book_titles, rotation=45, ha='right', fontsize=8)

    return bar_figure


def display_histogram(recommended_features, queried_book_features, book_titles, submission):
    similarities = cosine_similarity(
        queried_book_features, recommended_features).flatten()

    hist_figure, hist_axes = matplotlib.pyplot.subplots(figsize=(8, 6))
    hist_axes.hist(similarities, bins=10, color='skyblue',
                   edgecolor='black', alpha=0.7)

    hist_axes.set_xlabel('Similarity Score')
    hist_axes.set_ylabel('Number of Books')
    hist_axes.set_title('Distribution of Similarity Scores')

    return hist_figure


def open_book_link():
    try:
        selected_book = book_suggestions.set(
            book_suggestions.selection())['Title']
    except Exception as e:
        messagebox.showerror("Error", "A book must be selected first.")
        return
    page = requests.get(f'https://www.goodreads.com/search?q={selected_book}', headers={
                        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"})
    page_text = html.fromstring(page.content)
    book_link = page_text.xpath("//a[@class='bookTitle']/@href")[0]
    webbrowser.open(
        f'https://www.goodreads.com{book_link}', new=0, autoraise=True)


def embed_visualization(figure):
    if figure is not None:
        visualization_canvas = FigureCanvasTkAgg(
            figure, master=visualization_frame)
        visualization_canvas.draw()
        visualization_canvas.get_tk_widget().pack(fill="both", expand=True)
        matplotlib.pyplot.close(figure)
    stop_progress_bar()


def update_visualization():
    try:
        queried_book_title
    except:
        return
    clear_visualization()
    start_progress_bar()
    vis_type = selected_visualization.get()
    threading.Thread(target=generate_visualization, args=(vis_type,)).start()


def generate_visualization(vis_type):
    if vis_type == "Heatmap":
        figure = display_heatmap(
            recommended_features, queried_book_features, book_titles_with_query, queried_book_title)
    elif vis_type == "Bar Chart":
        figure = display_bar_chart(
            recommended_features, queried_book_features, book_titles_without_query, queried_book_title)
    elif vis_type == "Histogram":
        figure = display_histogram(
            recommended_features, queried_book_features, book_titles_with_query, queried_book_title)
    else:
        figure = None
    root.after(0, embed_visualization, figure)


def clear_visualization():
    for widget in visualization_frame.winfo_children():
        widget.destroy()


book_label = tk.Label(root, text="Book:")
book_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.E)
book_entry = tk.Entry(root, textvariable=book_var, width=50)
book_entry.grid(row=0, column=1, columnspan=2, padx=10, pady=5)
book_progress_bar = ttk.Progressbar(root, mode="determinate", length=200)
book_progress_bar.grid(row=0, column=3, padx=10, pady=5, sticky=tk.W)
scale_frame = tk.Frame(root)
scale_frame.grid(row=1, column=0, columnspan=5, sticky=tk.NSEW)
category_label = tk.Label(scale_frame, text="Category Weight")
category_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
category_weight = tk.DoubleVar(value=1.0)
category_weight_scale = ttk.Scale(
    scale_frame, length=140, from_=.1, to=2.0, variable=category_weight)
category_weight_scale.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
numerical_label = tk.Label(scale_frame, text="Numerical Weight")
numerical_label.grid(row=0, column=2, padx=5, pady=5)
numerical_weight = tk.DoubleVar(value=1.0)
numerical_weight_scale = ttk.Scale(
    scale_frame, length=140, from_=.1, to=2.0, variable=numerical_weight)
numerical_weight_scale.grid(row=0, column=3, padx=5, pady=5)
description_label = tk.Label(scale_frame, text="Description Weight")
description_label.grid(row=0, column=4, padx=5, pady=5, sticky=tk.E)
description_weight = tk.DoubleVar(value=1.0)
description_weight_scale = ttk.Scale(
    scale_frame, length=140, from_=.1, to=2.0, variable=description_weight)
description_weight_scale.grid(row=0, column=5, padx=5, pady=5, sticky=tk.E)
book_entry.bind('<Return>', lambda event: submit_with_loading())
book_submit = tk.Button(
    root, text="Recommend Similar Books", command=submit_with_loading)
book_submit.grid(row=2, column=0, columnspan=4)
book_suggestion_frame = tk.Frame(root)
book_suggestion_frame.grid(row=3, column=0, columnspan=4)
book_suggestions = ttk.Treeview(book_suggestion_frame, show="headings")
book_suggestions.grid(row=3, column=0,  sticky="nsew", pady=5, padx=(5, 0))
book_suggestions['columns'] = ('Title', 'Author', 'Score', 'Categories')
book_suggestions.heading('Title', text='Title')
book_suggestions.heading('Author', text='Author')
book_suggestions.heading('Score', text='Score')
book_suggestions.heading('Categories', text='Categories')
book_suggestions.column('Title', anchor=tk.CENTER, width=400)
book_suggestions.column('Author', anchor=tk.CENTER, width=150)
book_suggestions.column('Score', anchor=tk.CENTER, width=50, stretch=tk.NO)
book_suggestions.column('Categories', anchor=tk.CENTER, width=150)
book_suggestions.tag_configure('odd_row', background='white')
book_suggestions.tag_configure('even_row', background='#dadada')
suggestion_scrollbar = ttk.Scrollbar(
    book_suggestion_frame, orient="vertical", command=book_suggestions.yview)
suggestion_scrollbar.grid(row=3, column=1, sticky="nsew", pady=5, padx=(0, 5))
book_suggestions.configure(yscrollcommand=suggestion_scrollbar.set)
open_book_search = tk.Button(
    root, text="Find on Goodreads", command=open_book_link)
open_book_search.grid(row=4, column=0, columnspan=2, padx=10, pady=5)
visualization_options = ["Heatmap", "Bar Chart", "Histogram"]
selected_visualization = tk.StringVar(value=visualization_options[0])

visualization_label = tk.Label(root, text="Select Visualization:")
visualization_label.grid(row=4, column=2, padx=10, pady=5)

visualization_combobox = ttk.Combobox(
    root, textvariable=selected_visualization, values=visualization_options, state="readonly")
visualization_combobox.grid(row=4, column=3, padx=10, pady=5, sticky=tk.W)

visualization_combobox.bind("<<ComboboxSelected>>",
                            lambda event: update_visualization())

visualization_frame = tk.Frame(root)
visualization_frame.grid(row=5, column=0, columnspan=4, pady=10, sticky="nsew")


def main():
    root.mainloop()


if __name__ == "__main__":
    main()
