import './App.css';
import ImageUploader from 'react-images-upload';
import { useState, useEffect } from 'react';
import Button from 'react-bootstrap/Button'



function App() {
  const [picture, setPicture] = useState(0);

  const onDrop = picture => {
    setPicture(URL.createObjectURL(picture[picture.length-1]));
  };

  const handleClick = () => {
    const response = fetch("/hello_world")
    .then(response => response.json())
    .then(data => console.log(data.message))
  }

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
            <Button variant="primary" onClick={handleClick}>Submit</Button>{' '}

        </div>
      </div>
    );
}

function Palette() {

}

export default App;
