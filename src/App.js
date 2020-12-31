import './App.css';
import ImageUploader from 'react-images-upload';
import { useState, useEffect } from 'react';


function App() {
  const [picture, setPicture] = useState(0);

  const onDrop = picture => {
    setPicture(URL.createObjectURL(picture[picture.length-1]));
  };

    return (
      <div className="App">
      <div className="jumbotron">
        <h1 className="display-4"> Paint by Numbers </h1>
      </div>
      <div className="container">
        <ImageUploader
          withIcon={true}
          buttonText='Choose image'
          onChange={onDrop}
          imgExtension={['.jpg', '.gif', '.png']}
          maxFileSize={5242880}
        />
        { picture!==0 &&
          <div className="pictures">
            <img src={picture} width="200"/>
          </div>
        }
      </div>
      </div>
    );
}

function Palette() {

}

export default App;
