"""
Netflix Content Analysis — EDA
Run: python src/analysis.py --data_path data/netflix_titles_nov_2019.csv --output_dir outputs
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    """Load CSV and return DataFrame."""
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """Basic cleaning:
    - normalize missing country
    - explode multi-valued country and genres
    - parse date_added if present
    """
    df = df.copy()
    # Normalize country
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')
        df['country_list'] = df['country'].str.split(',')
        df = df.explode('country_list')
        df['country_list'] = df['country_list'].str.strip()
    # Normalize genres
    if 'listed_in' in df.columns:
        df['genre_list'] = df['listed_in'].fillna('').str.split(',')
        df = df.explode('genre_list')
        df['genre_list'] = df['genre_list'].str.strip()

    # Parse date_added
    if 'date_added' in df.columns:
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['added_month'] = df['date_added'].dt.month
        df['added_year'] = df['date_added'].dt.year

    return df


def plot_type_counts(df, output_dir):
    counts = df['type'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Movies vs TV Shows")
    ax.set_xlabel("Type")
    ax.set_ylabel("Count")
    plt.tight_layout()
    out = os.path.join(output_dir, "movies_vs_tv.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_top_countries(df, output_dir, top_n=10):
    # Use exploded country_list if available
    if 'country_list' in df.columns:
        country_counts = df['country_list'].value_counts().head(top_n)
    else:
        country_counts = df['country'].fillna('Unknown').str.split(',').explode().str.strip().value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(country_counts.index[::-1], country_counts.values[::-1])
    ax.set_title(f"Top {top_n} Countries Producing Netflix Content")
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Country")
    plt.tight_layout()
    out = os.path.join(output_dir, "top_countries.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_movies_vs_tv_by_country(df, output_dir, top_n=10):
    if 'country_list' not in df.columns:
        df = clean_data(df)
    ct = df.groupby(['country_list', 'type']).size().unstack(fill_value=0)
    top_countries = ct.sum(axis=1).sort_values(ascending=False).head(top_n).index
    plot_data = ct.loc[top_countries]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(plot_data))
    width = 0.35
    # bars
    ax.bar([i - width/2 for i in x], plot_data.get('Movie', 0), width, label='Movie')
    ax.bar([i + width/2 for i in x], plot_data.get('TV Show', 0), width, label='TV Show')
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_data.index, rotation=30, ha='right')
    ax.set_title(f"Movies vs TV Shows by Country (Top {top_n})")
    ax.set_xlabel("Country")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    out = os.path.join(output_dir, "movies_vs_tv_by_country.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_release_years(df, output_dir):
    if 'release_year' in df.columns:
        year_counts = df['release_year'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(year_counts.index, year_counts.values)
        ax.set_title("Number of Netflix Titles Released Each Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Titles")
        ax.grid(True)
        plt.tight_layout()
        out = os.path.join(output_dir, "release_years.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


def plot_genres(df, output_dir, top_n=15):
    if 'genre_list' not in df.columns:
        df = clean_data(df)
    genre_counts = df['genre_list'].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(genre_counts.index[::-1], genre_counts.values[::-1])
    ax.set_title(f"Top {top_n} Genres on Netflix")
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Genre")
    plt.tight_layout()
    out = os.path.join(output_dir, "top_genres.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main(data_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    df_clean = clean_data(df)

    # Generate plots
    plot_type_counts(df_clean, output_dir)
    plot_top_countries(df_clean, output_dir)
    plot_movies_vs_tv_by_country(df_clean, output_dir)
    plot_release_years(df_clean, output_dir)
    plot_genres(df_clean, output_dir)

    print("All plots saved to", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Netflix EDA")
    parser.add_argument("--data_path", type=str, default="data/netflix_titles_nov_2019.csv", help="Path to CSV")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Folder to save plots")
    args = parser.parse_args()
    main(args.data_path, args.output_dir)
