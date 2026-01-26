export interface MyDocument {
    _id: string;
    _source: {
        title: string;
        author: string;
        year: number;
        place: string;
        publisher: string;
        text: string;
    };
}

export interface Hit {
    _id: string;
    _source: {
        title: string;
        author: string;
        year: number;
        place: string;
    };
}

