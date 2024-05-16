import React, {useState} from 'react'  


export default function TextForm(props) {

  function handleclick(){
    console.log("uppercase was clicked"+text);
    setText(text.toUpperCase());
  }

  const handlechange=(event)=>{
    console.log("on change");
    setText(event.target.value);
  }

  const [text, setText] = useState('enter text here');

  return (
    <div>
        <h1>{props.heading}</h1>
        <div className="mb-3">
        <textarea className="form-control" value={text} onChange={handlechange} id="myBox" rows="8"></textarea>
        </div>
        <button className="btn btn-primary"  onClick={handleclick}>convert to upper case</button>
    </div>
  )
}
