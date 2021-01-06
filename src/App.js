import './App.css';
import ImageUploader from 'react-images-upload';
import { useState, useEffect } from 'react';
import Button from 'react-bootstrap/Button'


function App() {
  const [picture, setPicture] = useState(0);
  const [processedImage, setProcessedImage] = useState(0);
  const [f, setF] = useState(0);

  const onDrop = p => {
    console.log("p", p)
    setF(p[p.length-1]);
    console.log("f", f);
    setPicture(URL.createObjectURL(p[p.length-1]));

  };

  const handleClick_ = () => {
    fetch("/hello_world")
    .then(response => response.json())
    .then(data => console.log(data.message))
  };

  const handleReset = () => {};

  const handleSubmit = async function() {
    const data = new FormData();
    data.append('file', f);

    fetch("/get_palette", { 
      method: "POST",
      body: data,
    })
    .then(response => {
      return response.blob();
    })
    .then(img => {
      console.log(img);
      setProcessedImage(URL.createObjectURL(img));
    });
//    .then(_ => fetch("/get_outline"))
//    .then(response => console.log(response.json()));;

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
            <div className="pictures">
              <div className="picture">
                { picture!==0 && <img src={picture} width="300"/> }
              </div>
              <div className="picture">
                { processedImage !==0 && <img src={processedImage} width="300"/> }
              </div>


            </div>
            <div className="buttons">
                <div className="button">
                    { picture!==0 && <Button variant="primary" onClick={handleSubmit}>Submit</Button> }
                </div>
                <div className="button">
                    { picture!==0 && <Button variant="primary" onClick={handleReset}>Reset</Button> }
                </div>
             </div>

        </div>
      </div>
    );
}

function Palette() {

}

export default App;
