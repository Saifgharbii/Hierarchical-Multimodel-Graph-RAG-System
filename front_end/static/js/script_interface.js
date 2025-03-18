// Fonctionnalité drag-and-drop pour les fichiers et images
document.addEventListener('DOMContentLoaded', function() {
    const fileUploadBtn = document.querySelector('button[title="Attach file"]');
    const imageUploadBtn = document.querySelector('button[title="Upload image"]');
    const userInput = document.getElementById('userInput');
    const chatForm = document.getElementById('chatForm');
    const dropZone = chatForm; // Utiliser le formulaire comme zone de drop
    
    // Créer les inputs de fichier cachés
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.id = 'fileInput';
    fileInput.style.display = 'none';
    fileInput.accept = '.pdf,.doc,.docx,.txt,.csv,.xls,.xlsx';
    
    const imageInput = document.createElement('input');
    imageInput.type = 'file';
    imageInput.id = 'imageInput';
    imageInput.style.display = 'none';
    imageInput.accept = 'image/*';
    
    document.body.appendChild(fileInput);
    document.body.appendChild(imageInput);
    
    // Événements pour les boutons d'upload
    fileUploadBtn.addEventListener('click', function() {
        fileInput.click();
    });
    
    imageUploadBtn.addEventListener('click', function() {
        imageInput.click();
    });
    
    // Gérer la sélection de fichier
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
    
    imageInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0], true);
        }
    });
    
    // Configurer le drag-and-drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropZone.classList.add('drag-highlight');
    }
    
    function unhighlight() {
        dropZone.classList.remove('drag-highlight');
    }
    
    // Gérer le drop des fichiers
    dropZone.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }
    
    // Fonction pour traiter les fichiers
    function handleFile(file, isImage = false) {
        // Enregistrer la référence au fichier
        const fileData = {
            name: file.name,
            size: formatFileSize(file.size),
            type: file.type,
            isImage: isImage || file.type.startsWith('image/')
        };
        
        // Stocker les données du fichier pour la soumission
        chatForm.dataset.fileName = file.name;
        chatForm.dataset.fileSize = fileData.size;
        chatForm.dataset.fileType = file.type;
        chatForm.dataset.isImage = fileData.isImage;
        
        // Créer la prévisualisation du fichier
        createFilePreview(fileData);
        
        // Changer l'état du bouton correspondant
        if (fileData.isImage) {
            imageUploadBtn.classList.add('active');
        } else {
            fileUploadBtn.classList.add('active');
        }
    }
    
    // Fonction pour formater la taille du fichier
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Fonction pour afficher la prévisualisation du fichier
    function createFilePreview(fileData) {
        // Supprimer toute prévisualisation existante
        const existingPreview = document.getElementById('file-preview');
        if (existingPreview) {
            existingPreview.remove();
        }
        
        // Créer la prévisualisation
        const previewContainer = document.createElement('div');
        previewContainer.id = 'file-preview';
        previewContainer.className = 'file-preview';
        
        // Choisir l'icône appropriée
        let fileIcon = 'bi-file-earmark';
        if (fileData.isImage) {
            fileIcon = 'bi-file-earmark-image';
        } else if (fileData.type.includes('pdf')) {
            fileIcon = 'bi-file-earmark-pdf';
        } else if (fileData.type.includes('word') || fileData.type.includes('doc')) {
            fileIcon = 'bi-file-earmark-word';
        } else if (fileData.type.includes('excel') || fileData.type.includes('sheet') || fileData.type.includes('csv')) {
            fileIcon = 'bi-file-earmark-spreadsheet';
        } else if (fileData.type.includes('text')) {
            fileIcon = 'bi-file-earmark-text';
        }
        
        previewContainer.innerHTML = `
            <div class="preview-content">
                <i class="bi ${fileIcon}"></i>
                <div class="file-info">
                    <div class="file-name">${fileData.name}</div>
                    <div class="file-size">${fileData.size}</div>
                </div>
            </div>
            <button type="button" class="remove-file" title="Remove file">
                <i class="bi bi-x"></i>
            </button>
        `;
        
        // Ajouter la prévisualisation juste au-dessus du champ de saisie
        chatForm.insertBefore(previewContainer, chatForm.firstChild);
        
        // Ajouter un événement pour supprimer le fichier
        previewContainer.querySelector('.remove-file').addEventListener('click', function() {
            previewContainer.remove();
            delete chatForm.dataset.fileName;
            delete chatForm.dataset.fileSize;
            delete chatForm.dataset.fileType;
            delete chatForm.dataset.isImage;
            fileUploadBtn.classList.remove('active');
            imageUploadBtn.classList.remove('active');
        });
    }
    
    // Gérer la soumission du formulaire
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const messageText = userInput.value.trim();
        const fileName = chatForm.dataset.fileName;
        const fileSize = chatForm.dataset.fileSize;
        const fileType = chatForm.dataset.fileType;
        const isImage = chatForm.dataset.isImage === 'true';
        
        if (messageText || fileName) {
            // Ajouter le message à l'interface
            addMessageToChat(messageText, {
                name: fileName,
                size: fileSize,
                type: fileType,
                isImage: isImage
            });
            
            // Réinitialiser le formulaire
            userInput.value = '';
            delete chatForm.dataset.fileName;
            delete chatForm.dataset.fileSize;
            delete chatForm.dataset.fileType;
            delete chatForm.dataset.isImage;
            
            // Supprimer la prévisualisation
            const filePreview = document.getElementById('file-preview');
            if (filePreview) {
                filePreview.remove();
            }
            
            // Réinitialiser les styles des boutons
            fileUploadBtn.classList.remove('active');
            imageUploadBtn.classList.remove('active');
        }
    });
    
    // Fonction pour ajouter des messages au chat
    function addMessageToChat(messageText, fileData) {
        const messagesContainer = document.getElementById('messagesContainer');
        
        // Créer la rangée de message
        const messageRow = document.createElement('div');
        messageRow.className = 'message-row';
        
        let messageContent = '<div class="user-message">';
        
        // Ajouter le texte du message s'il est présent
        if (messageText) {
            messageContent += `<p>${messageText}</p>`;
        }
        
        // Ajouter la pièce jointe si présente
        if (fileData && fileData.name) {
            // Choisir l'icône appropriée
            let fileIcon = 'bi-file-earmark';
            if (fileData.isImage) {
                fileIcon = 'bi-file-earmark-image';
            } else if (fileData.type && fileData.type.includes('pdf')) {
                fileIcon = 'bi-file-earmark-pdf';
            } else if (fileData.type && (fileData.type.includes('word') || fileData.type.includes('doc'))) {
                fileIcon = 'bi-file-earmark-word';
            } else if (fileData.type && (fileData.type.includes('excel') || fileData.type.includes('sheet') || fileData.type.includes('csv'))) {
                fileIcon = 'bi-file-earmark-spreadsheet';
            } else if (fileData.type && fileData.type.includes('text')) {
                fileIcon = 'bi-file-earmark-text';
            }
            
            messageContent += `
                <div class="attached-file">
                    <i class="bi ${fileIcon}"></i>
                    <div class="file-info">
                        <div class="file-name">${fileData.name}</div>
                        <div class="file-size">${fileData.size || ''}</div>
                    </div>
                </div>
            `;
        }
        
        messageContent += '</div>';
        messageRow.innerHTML = messageContent;
        messagesContainer.appendChild(messageRow);
        
        // Faire défiler vers le bas du chat
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Code pour simuler une réponse (à modifier selon votre logique)
        setTimeout(function() {
            const assistantRow = document.createElement('div');
            assistantRow.className = 'message-row assistant-message';
            assistantRow.innerHTML = `
                <div class="assistant-icon">
                    <i class="bi bi-robot"></i>
                </div>
                <div class="assistant-bubble">
                    J'ai bien reçu votre message${fileData && fileData.name ? ' et votre fichier' : ''}.
                </div>
            `;
            messagesContainer.appendChild(assistantRow);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }, 1000);
    }
});
